import numpy as np
from qiskit import Aer, transpile

from cost_sections import InversionGeneral

class SequentialVCMinimizer:
    def __init__(
        self,
        ansatz,
        cost_function,
        backend,
        default_section=InversionGeneral,
        error_mitigation_filter=None,
        transpile_options={'optimization_level': 1},
        shots=8192,
        run_options={},
        num_experiments=1,
        seed=12345
    ):
        self.ansatz = ansatz.copy()
        
        self.cost_function = cost_function
        
        self.cost_section_generators = [default_section for _ in range(len(self.ansatz.parameters))]

        self.backend = backend
        self.num_experiments = num_experiments
        self.shots = shots
        self.transpile_options = dict(transpile_options)
        self.run_options = dict(run_options)
        
        self.error_mitigation_filter = error_mitigation_filter
        
        self.random_gen = np.random.default_rng(seed)
        
        self.statevector_simulator = None
        self.ideal_costs_step = None
        self.ideal_costs_sweep = None
        
    def minimize(
        self,
        initial_param_val,
        max_sweeps=20,
        callbacks_step=[],
        callbacks_sweep=[],
        convergence_distance=1.e-2,
        convergence_cost=1.e-4,
        gradient_descent_threshold=0.
    ):
        num_params = len(self.ansatz.parameters)
        assert len(initial_param_val) == num_params
        
        param_val = np.array(initial_param_val)

        cost_values = []
        nexp = 0

        sweep_param_val = param_val.copy()
        gradient_descent = False
        mean_cost = 1.
        
        for isweep in range(max_sweeps):
            if not gradient_descent and gradient_descent_threshold > 0. and mean_cost < gradient_descent_threshold:
                print('Switching to gradient descent at parameter values', '[' + ', '.join(map(str, sweep_param_val)) + ']')
                gradient_descent = True
            
            if gradient_descent:
                cost_sections = []
                for iparam in range(num_params):
                    cost_section, costs = self._smo_one_iter(sweep_param_val, iparam)
                    cost_sections.append(cost_section)
                    nexp += len(costs) * self.num_experiments
                    
                gradient = np.array([c.grad(c.current) for c in cost_sections])
                steepest_direction = np.argmax(np.abs(gradient))
                step_size_steepest = (cost_sections[steepest_direction].minimize() - sweep_param_val[steepest_direction]) * 2. / 3.
                learning_rate = -step_size_steepest / gradient[steepest_direction]
                
                sweep_param_val += -gradient * learning_rate
                
                cost_values.extend([c.fun(theta_updated) for c, theta_updated in zip(cost_sections, sweep_param_val)])
                
            else:
                for iparam in range(num_params):
                    cost_section, costs = self._smo_one_iter(sweep_param_val, iparam)

                    theta_opt = cost_section.minimize()

                    for callback in callbacks_step:
                        callback(isweep, iparam, sweep_param_val, theta_opt, costs, cost_section)

                    sweep_param_val[iparam] = theta_opt
                    nexp += len(costs) * self.num_experiments

                    cost_value = cost_section.fun(theta_opt)
                    cost_values.append(cost_value)

            sweep_mean_cost = np.mean(cost_values[-num_params:])
            
            for callback in callbacks_sweep:
                callback(isweep, param_val, sweep_param_val, mean_cost, sweep_mean_cost)
                
            distance = np.max(np.abs(param_val - sweep_param_val))
            
            if distance < convergence_distance:
                print('SMO converged by parameter distance')
                break
                
            cost_update = sweep_mean_cost - mean_cost
            
            if abs(cost_update) < convergence_cost:
                print('SMO converged by cost update')
                break
                
            if not gradient_descent and gradient_descent_threshold < 0. and cost_update < 0. and cost_update > gradient_descent_threshold:
                print('Switching to gradient descent at parameter values', '[' + ', '.join(map(str, sweep_param_val)) + ']')
                gradient_descent = True
                
            mean_cost = sweep_mean_cost
            param_val = sweep_param_val.copy()

        return param_val, cost_values, nexp
    
    def ideal_cost_step(self, isweep, iparam, sweep_param_val, theta_opt, costs, cost_section):
        """Convenience function for tracking the ideal cost through callback
        """
        if self.statevector_simulator is None:
            self.statevector_simulator = Aer.get_backend('statevector_simulator')

        if self.ideal_costs_step is None:
            self.ideal_costs_step = []
            
        circuit = transpile(self._make_circuit(sweep_param_val, iparam, theta_opt), backend=self.statevector_simulator)
        prob_dist = np.square(np.abs(self.statevector_simulator.run(circuit).result().data()['statevector']))
        cost = self.cost_function(prob_dist)
        
        self.ideal_costs_step.append(cost)
        
    def ideal_cost_sweep(self, isweep, param_val, sweep_param_val):
        """Convenience function for tracking the ideal cost through callback
        """
        if self.statevector_simulator is None:
            self.statevector_simulator = Aer.get_backend('statevector_simulator')

        if self.ideal_costs_sweep is None:
            self.ideal_costs_sweep = []
            
        circuit = transpile(self._make_circuit(sweep_param_val), backend=self.statevector_simulator)
        prob_dist = np.square(np.abs(self.statevector_simulator.run(circuit).result().data()['statevector']))
        cost = self.cost_function(prob_dist)
        
        self.ideal_costs_sweep.append(cost)
    
    def _make_circuit(self, param_val, param_id=None, test_value=None):
        if param_id is not None:
            param_val = np.copy(param_val)
            param_val[param_id] = test_value
            
        param_dict = dict(zip(self.ansatz.parameters, param_val))
        
        return self.ansatz.bind_parameters(param_dict)
    
    def _run_circuits(self, circuits):
        if self.backend.name() != 'statevector_simulator':
            for circuit in circuits:
                circuit.measure(circuit.qregs[0], circuit.cregs[0])
        
        if self.backend.configuration().simulator:
            circuits = transpile(circuits, backend=self.backend, **self.transpile_options)
        else:
            circuits = transpile_with_dynamical_decoupling(circuits, backend=self.backend, **self.transpile_options)
            
        if self.backend.name() == 'statevector_simulator':
            result = self.backend.run(circuits).result()
            
            total_shots = self.shots * self.num_experiments
            
            probs = []
            for res in result.results:
                exact = np.square(np.abs(res.data.statevector))
                if total_shots <= 0:
                    probs.append(exact)
                else:
                    counts = self.random_gen.multinomial(total_shots, exact)
                    probs.append(counts / total_shots)
                
        else:
            circuits_to_run = []
            for circuit in circuits:
                circuits_to_run += [circuit] * self.num_experiments

            raw_counts = self.backend.run(circuits_to_run, shots=self.shots).result().get_counts()
            if len(circuits_to_run) == 1:
                num_circuits = 1
                counts_list = [raw_counts]
            else:
                num_circuits = len(raw_counts) // self.num_experiments
                counts_list = [dict() for _ in range(num_circuits)]

                for iexp, cdict in enumerate(raw_counts):
                    ic = iexp // self.num_experiments
                    for key, value in cdict.items():
                        try:
                            counts_list[ic][key] += value
                        except KeyError:
                            counts_list[ic][key] = value

            if self.error_mitigation_filter is not None:
                corrected_counts_list = []
                for counts in counts_list:
                    corrected_counts_list.append(self.error_mitigation_filter.apply(counts))

                counts_list = corrected_counts_list

            probs = [np.zeros(2 ** self.ansatz.num_qubits, dtype='f8') for _ in range(num_circuits)]
            for ic, counts in enumerate(counts_list):
                total = sum(counts.values())
                for key, value in counts.items():
                    probs[ic][int(key, 2)] = value / total
                
        return probs
        
    def _compute_costs(self, circuits, num_toys=None):
        prob_dists = self._run_circuits(circuits)
        
        costs = np.empty(len(prob_dists), dtype='f8')
        for iprob, prob_dist in enumerate(prob_dists):
            costs[iprob] = self.cost_function(prob_dist)
            
        if not num_toys:
            return costs
        
        shots = self.shots * self.num_experiments
        sigmas = np.empty_like(costs)
        
        for iprob, prob_dist in enumerate(prob_dists):
            sumw = 0.
            sumw2 = 0.
            for _ in range(num_toys):
                toy_prob_dist = self.random_gen.multinomial(shots, prob_dist) / shots
                toy_cost = self.cost_function(toy_prob_dist)
                sumw += toy_cost
                sumw2 += toy_cost * toy_cost
            
            mean = sumw / num_toys
            sigmas[iprob] = np.sqrt(sumw2 / num_toys - mean * mean)
            
        return costs, sigmas
    
    def _smo_one_iter(
        self,
        param_val,
        param_id
    ):
        current = param_val[param_id]
        
        cost_section = self.cost_section_generators[param_id]()
        cost_section.set_thetas(current)
        
        circuits = []
        for theta in cost_section.thetas:
            circuits.append(self._make_circuit(param_val, param_id, theta))
        
        costs = self._compute_costs(circuits)
        
        cost_section.set_coeffs(costs)

        return cost_section, costs