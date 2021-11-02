import logging
import numpy as np
from qiskit import Aer, transpile
from qiskit.result.counts import Counts

from cost_sections import InversionGeneral

logger = logging.getLogger('four_qubit_schwinger.sequential_minimizer')

def combine_counts(new, base):
    data = dict(base)
    for key, value in new.items():
        try:
            data[key] += value
        except KeyError:
            data[key] = value
                
    return Counts(data, time_taken=base.time_taken, creg_sizes=base.creg_sizes, memory_slots=base.memory_slots)

class SequentialVCMinimizer:
    def __init__(
        self,
        ansatz,
        cost_function,
        backend,
        default_section=InversionGeneral,
        strategy='sequential',
        error_mitigation_filter=None,
        transpile_options={'optimization_level': 1},
        shots=8192,
        run_options={},
        seed=12345
    ):
        self.ansatz = ansatz.copy()
        
        self.cost_function = cost_function
        
        self.cost_section_generators = [default_section for _ in range(len(self.ansatz.parameters))]
        
        self.strategy = strategy

        self.backend = backend
        self.shots = shots
        self.transpile_options = dict(transpile_options)
        self.run_options = dict(run_options)
        
        self.error_mitigation_filter = error_mitigation_filter
        
        self.random_gen = np.random.default_rng(seed)
        
        self.callbacks_step = []
        self.callbacks_sweep = []
        
        self.statevector_simulator = None
        self.ideal_costs_step = None
        self.shots_step = None
        self.ideal_costs_sweep = None
        self.shots_sweep = None
        
    def minimize(
        self,
        initial_param_val,
        max_sweeps=40,
        convergence_distance=1.e-3,
        convergence_cost=1.e-4
    ):
        assert len(initial_param_val) == len(self.ansatz.parameters)
        
        logger.info('Starting minimize() with initial parameters {}'.format(str(initial_param_val)))

        # Returned values
        param_val = np.array(initial_param_val)
        total_shots = 0

        # Values to be updated after each sweep
        current_cost = 1.
        if self.strategy == 'largest-drop':
            self.no_scouting_direction = -1
        
        for isweep in range(max_sweeps):
            logger.info('Sweep {} with strategy {} - current cost {} current shots {}'.format(isweep, self.strategy, current_cost, total_shots))
            
            if self.strategy != 'largest_drop':
                self.no_scouting_direction = -1
            
            if self.strategy == 'sequential':
                sweep_res = self._minimize_sequential(isweep, param_val, current_cost, total_shots)

            elif self.strategy == 'gradient-descent':
                sweep_res = self._minimize_gradient_descent(isweep, param_val, current_cost, total_shots)
                
            elif self.strategy == 'largest-drop':
                sweep_res = self._minimize_largest_drop(isweep, param_val, current_cost, total_shots)
                
            sweep_param_val, sweep_cost, sweep_shots = sweep_res
                    
            callback_arg = {
                'isweep': isweep,
                'current_param_val': param_val,
                'sweep_param_val': sweep_param_val,
                'current_cost': current_cost,
                'sweep_cost': sweep_cost,
                'current_shots': total_shots,
                'sweep_shots': sweep_shots
            }
            for callback in self.callbacks_sweep:
                callback(self, callback_arg)
                
            distance = np.max(np.abs(param_val - sweep_param_val))
            
            if distance < convergence_distance:
                logger.info('Minimization converged by parameter distance')
                break
                
            cost_update = sweep_cost - current_cost
            
            if abs(cost_update) < convergence_cost:
                logger.info('Minimization converged by cost update')
                break
                
            param_val = sweep_param_val
            total_shots += sweep_shots
            current_cost = sweep_cost

        return param_val, total_shots
    
    def _minimize_sequential(self, isweep, param_val, current_cost, total_shots):
        sweep_param_val = param_val.copy()
        sweep_shots = 0
        
        for iparam in range(len(self.ansatz.parameters)):
            logger.debug('sequential: Calculating cost section for parameter {}'.format(iparam))
            cost_section, costs, shots = self._calculate_cost_section(sweep_param_val, iparam)

            theta_opt = cost_section.minimum()
            
            callback_arg = {
                'isweep': isweep,
                'iparam': iparam,
                'param_val': sweep_param_val,
                'theta_opt': theta_opt,
                'costs': costs,
                'cost_section': cost_section,
                'current_shots': total_shots + sweep_shots,
                'step_shots': shots
            }
            for callback in self.callbacks_step:
                callback(self, callback_arg)

            sweep_param_val[iparam] = theta_opt
            sweep_shots += shots

        sweep_cost = cost_section.fun(theta_opt)

        return sweep_param_val, sweep_cost, sweep_shots
    
    def _minimize_gradient_descent(self, isweep, param_val, current_cost, total_shots):
        sweep_shots = 0
        raw_data = dict()
        
        param_ids = list(range(len(self.ansatz.parameters)))
        
        logger.info('gradient descent: Calculating cost sections for all parameters')
        cost_sections, costs, shots = self._calculate_cost_section(param_val, param_ids, raw_data=raw_data, reuse=True)

        if len(self.callbacks_step) != 0:
            for iparam in param_ids:
                callback_arg = {
                    'isweep': isweep,
                    'iparam': iparam,
                    'param_val': param_val,
                    'theta_opt': cost_sections[iparam].minimum(),
                    'costs': costs[iparam],
                    'cost_section': cost_sections[iparam],
                    'current_shots': total_shots + sweep_shots,
                    'step_shots': shots
                }
                for callback in self.callbacks_step:
                    callback(self, callback_arg)            

        sweep_shots += shots

        gradient = np.array([cost_sections[ip].grad() for ip in param_ids])
        steepest_direction = np.argmax(np.abs(gradient))
        step_size_steepest = (cost_sections[steepest_direction].minimum() - param_val[steepest_direction]) * 1. / 3.
        learning_rate = step_size_steepest / (-gradient[steepest_direction])
        parameter_shift = -gradient * learning_rate

        sweep_param_val = param_val + parameter_shift
        sweep_cost = current_cost - gradient @ parameter_shift # linear approximation
        
        return sweep_param_val, sweep_cost, sweep_shots
    
    def _minimize_largest_drop(self, isweep, param_val, current_cost, total_shots):
        sweep_shots = 0
        
        ## Scouting run
        shots_original = self.shots
        self.shots //= 16

        raw_data = dict()
        
        param_ids = list(range(len(self.ansatz.parameters)))
        
        logger.info('largest drop: Calculating cost sections for all parameters')

        try:
            param_ids.remove(self.no_scouting_direction)
        except ValueError:
            pass
        
        cost_sections, _, shots = self._calculate_cost_section(param_val, param_ids, raw_data=raw_data, reuse=True)
        
        sweep_shots += shots
        
        largest_drop = 0.
        step_direction = -1
        
        for iparam in param_ids:
            drop = cost_sections[iparam].fun() - cost_sections[iparam].fun(cost_sections[iparam].minimum())
            if drop > largest_drop:
                step_direction = iparam
                largest_drop = drop

        self.shots = shots_original

        if largest_drop == 0.:
            return param_val, current_cost, sweep_shots
        
        logger.info('largest drop: Stepping in direction {}'.format(step_direction))
        
        cost_section, costs, shots = self._calculate_cost_section(param_val, step_direction, raw_data=raw_data, reuse=False)

        theta_opt = cost_section.minimum()

        callback_arg = {
            'isweep': isweep,
            'iparam': step_direction,
            'param_val': param_val,
            'theta_opt': theta_opt,
            'costs': costs,
            'cost_section': cost_section,
            'current_shots': total_shots + sweep_shots,
            'step_shots': shots
        }
        for callback in self.callbacks_step:
            callback(self, callback_arg)

        sweep_param_val = param_val.copy()
        sweep_param_val[step_direction] = theta_opt
        sweep_cost = cost_section.fun(theta_opt)

        sweep_shots += shots

        self.no_scouting_direction = step_direction
        
        return sweep_param_val, sweep_cost, sweep_shots
    
    def ideal_cost_step(self, arg):
        """Convenience function for tracking the ideal cost through callback
        """
        if self.statevector_simulator is None:
            self.statevector_simulator = Aer.get_backend('statevector_simulator')

        if self.ideal_costs_step is None:
            self.ideal_costs_step = []
            self.shots_step = []
            
        circuit = transpile(self._make_circuit(arg['param_val'], arg['iparam'], arg['theta_opt']), backend=self.statevector_simulator)
        prob_dist = np.square(np.abs(self.statevector_simulator.run(circuit).result().data()['statevector']))
        cost = self.cost_function(prob_dist)
        
        self.ideal_costs_step.append(cost)
        self.shots_step.append(arg['current_shots'] + arg['step_shots'])
        
    def ideal_cost_sweep(self, arg):
        """Convenience function for tracking the ideal cost through callback
        """
        if self.statevector_simulator is None:
            self.statevector_simulator = Aer.get_backend('statevector_simulator')

        if self.ideal_costs_sweep is None:
            self.ideal_costs_sweep = []
            self.shots_sweep = []
            
        circuit = transpile(self._make_circuit(arg['sweep_param_val']), backend=self.statevector_simulator)
        prob_dist = np.square(np.abs(self.statevector_simulator.run(circuit).result().data()['statevector']))
        cost = self.cost_function(prob_dist)
        
        self.ideal_costs_sweep.append(cost)
        self.shots_sweep.append(arg['current_shots'] + arg['sweep_shots'])
    
    def _make_circuit(self, param_val, param_id=None, test_value=None):
        if param_id is not None:
            param_val = np.copy(param_val)
            param_val[param_id] = test_value
            
        param_dict = dict(zip(self.ansatz.parameters, param_val))
        
        circuit = self.ansatz.bind_parameters(param_dict)
        if self.backend.name() != 'statevector_simulator':
            circuit.measure_all(inplace=True)
            
        return circuit
    
    def _run_circuits(self, circuits):
        """Run the circuits
        Args:
            circuits (list): List of circuits or dicts. Dict elements represent raw data from previous experiments.
        """
        if self.backend.configuration().simulator:
            circuits_to_run = transpile(circuits, backend=self.backend, **self.transpile_options)
        else:
            circuits_to_run = transpile_with_dynamical_decoupling(circuits, backend=self.backend, **self.transpile_options)
            
        if self.backend.name() == 'statevector_simulator':
            run_options = self.run_options
            shots_used = self.shots * len(circuits_to_run) # dummy
        else:
            run_options = dict(self.run_options)
            max_shots = self.backend.configuration().max_shots
            if max_shots < self.shots:
                run_options['shots'] = max_shots
                circuits_to_run *= (self.shots // max_shots)
            else:
                run_options['shots'] = self.shots
                
            shots_used = run_options['shots'] * len(circuits_to_run)
            
        logger.info('run circuits: Running {} experiments, {} shots each'.format(len(circuits_to_run), run_options['shots']))
        
        max_experiments = backend.configuration().max_experiments
        circuit_blocks = []
        for ic in range(0, len(circuits_to_run), max_experiments):
            circuit_blocks.append(circuits_to_run[ic:ic + max_experiments])
            
        results = list(Counts(dict()) for _ in range(len(circuits)))
        ires = 0
        for circuit_block in circuit_blocks:
            run_result = self.backend.run(circuit_block, **run_options).result()

            if self.backend.name() == 'statevector_simulator':
                results += [res.data.statevector for res in run_result.results]
            else:
                counts_list = run_result.get_counts()
                for counts in counts_list:
                    results[ires] = combine_counts(results[ires], counts)
                    ires = (ires + 1) % len(circuits)
            
        return results, shots_used
    
    def _compute_probabilities(self, results):
        prob_dists = []
        
        if self.backend.name() == 'statevector_simulator':
            for statevector in results:
                exact = np.square(np.abs(statevector))
                if self.shots <= 0:
                    prob_dists.append(exact)
                else:
                    prob_dist = self.random_gen.multinomial(self.shots, exact) / self.shots
                    prob_dists.append(prob_dist)
                
        else:
            if self.error_mitigation_filter is not None:
                corrected_counts = []
                for counts in results:
                    counts_dict = self.error_mitigation_filter.apply(counts)
                    corrected_counts.append(Counts(counts_dict, time_taken=counts.time_taken, creg_sizes=counts.creg_sizes, memory_slots=counts.memory_slots))
            else:
                corrected_counts = results

            for counts in corrected_counts:
                total = sum(counts.values())
                prob_dist = np.zeros(2 ** self.ansatz.num_qubits, dtype='f8')
                for idx, value in counts.int_outcomes().items():
                    prob_dist[idx] = value / total

                prob_dists.append(prob_dist)
                
        return prob_dists
    
    def _compute_costs(self, prob_dists, num_toys=0):
        costs = np.empty(len(prob_dists), dtype='f8')
        for iprob, prob_dist in enumerate(prob_dists):
            costs[iprob] = self.cost_function(prob_dist)
            
        if num_toys == 0:
            return costs
        
        sigmas = np.empty_like(costs)
        
        for iprob, prob_dist in enumerate(prob_dists):
            sumw = 0.
            sumw2 = 0.
            for _ in range(num_toys):
                toy_prob_dist = self.random_gen.multinomial(self.shots, prob_dist) / self.shots
                toy_cost = self.cost_function(toy_prob_dist)
                sumw += toy_cost
                sumw2 += toy_cost * toy_cost
            
            mean = sumw / num_toys
            sigmas[iprob] = np.sqrt(sumw2 / num_toys - mean * mean)
            
        return costs, sigmas
    
    def _calculate_cost_section(
        self,
        param_val,
        param_ids,
        raw_data=None,
        reuse=False
    ):
        """Instantiate a cost section object and calculate the 1D section function from cost measurements.

        Args:
            param_val (ndarray): Current parameter vector.
            param_ids (int or list): Index of the parameter to compute the section over.
            raw_data (dict): Map of parameter values to raw experiment results (counts dict or statevector)
            reuse (bool): If True, raw_data is used instead of performing new measurements, if available.
                If False, new measurements are performed and raw_data is updated (applicable only to counts)
        """
        if isinstance(param_ids, int):
            list_input = False
            param_ids = [param_ids]
        else:
            list_input = True
            
        circuits = []
        circuit_param_ids = []

        cost_sections = dict()
        reused_results = dict()

        for param_id in param_ids:
            cost_section = self.cost_section_generators[param_id]()
            cost_section.set_thetas(param_val[param_id])
            
            cost_sections[param_id] = cost_section
            reused_results[param_id] = []

            for itheta, theta in enumerate(cost_section.thetas):
                if raw_data and reuse:
                    key = tuple(np.concatenate((param_val[:param_id], [theta], param_val[param_id + 1:])))
                    try:
                        reused_results[param_id].append((itheta, raw_data[key]))
                        continue
                    except KeyError:
                        pass

                circuits.append(self._make_circuit(param_val, param_id, theta))
                circuit_param_ids.append(param_id)
                
        results, shots = self._run_circuits(circuits)
        
        results_per_param = dict((param_id, list()) for param_id in param_ids)
        for param_id, result in zip(circuit_param_ids, results):
            results_per_param[param_id].append(result)
            
        costs_per_param = dict()
            
        for param_id in param_ids:
            results = results_per_param[param_id]
            cost_section = cost_sections[param_id]

            for index, result in reused_results[param_id]:
                results.insert(index, result)

            if raw_data:
                for theta, result in zip(cost_section.thetas, results):
                    key = tuple(np.concatenate((param_val[:param_id], [theta], param_val[param_id + 1:])))

                    if self.backend.name() == 'statevector_simulator':
                        raw_data[key] = result
                    else:
                        try:
                            base = raw_data[key]
                        except KeyError:
                            base = Counts(dict())

                        raw_data[key] = combine_counts(result, base)

            prob_dists = self._compute_probabilities(results)

            costs = self._compute_costs(prob_dists)

            cost_section.set_coeffs(costs)
            
            costs_per_param[param_id] = costs
            
        if list_input:
            return cost_sections, costs_per_param, shots
        else:
            param_id = param_ids[0]
            return cost_sections[param_id], costs_per_param[param_id], shots
    

class SectionGenSwitcher:
    def __init__(self, generator, param_ids, threshold):
        self.generator = generator
        self.param_ids = param_ids
        self.threshold = threshold
        self.switched = False
        
    def callback_sweep(self, minimizer, args):
        if self.switched or args['sweep_cost'] > self.threshold:
            return
        
        for iparam in self.param_ids:
            minimizer.cost_section_generators[iparam] = self.generator

        self.switched = True
