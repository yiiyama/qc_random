from typing import Tuple
import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scistats
from qiskit import Aer, QuantumCircuit, QuantumRegister, converters, transpile
from qiskit.circuit import ParameterVector

#####################################################################################################
### ANSATZ ##########################################################################################
#####################################################################################################

def make_Agate(theta, phi):
    qc_Agate = QuantumCircuit(2, name = 'Agate')

    qc_Agate.cx(1,0)
    qc_Agate.rz(-phi,1)
    qc_Agate.ry(-theta,1)
    qc_Agate.cx(0,1)
    qc_Agate.ry(theta,1)
    qc_Agate.rz(phi,1)
    qc_Agate.cx(1,0)

    return qc_Agate

def make_pnp(
    num_qubit : int, 
    num_layer : int,
    is_barrier : bool = False
):
    num_parameter = (num_qubit - 1) * num_layer * 2
    pv = ParameterVector('\N{greek small letter theta}', num_parameter)

    qr = QuantumRegister(num_qubit, 'q')
    qc_pnp  = QuantumCircuit(qr)
    index_pv = 0
        
    for i_layer in range(num_layer):
        for qi in range(0, num_qubit-1, 2):
            A_gate_inst = make_Agate(pv[index_pv], pv[index_pv+1]).to_instruction()
            index_pv += 2
            qc_pnp.append(A_gate_inst, [qr[qi], qr[qi+1]])

        for qi in range(1, num_qubit - 1, 2):
            A_gate_inst = make_Agate(pv[index_pv], pv[index_pv+1]).to_instruction()
            index_pv += 2
            qc_pnp.append(A_gate_inst, [qr[qi], qr[qi+1]])

        if is_barrier:
            qc_pnp.barrier()

    return qc_pnp, pv

def make_ansatz(
    num_particle : int, 
    num_qubit : int, 
    num_layer : int,
    is_barrier : bool = False
):
    qc_pnp, pv = make_pnp(num_qubit, num_layer, is_barrier)
    qc_ansatz = QuantumCircuit(num_qubit)

    for pi in range(num_particle):
        if 2 * pi < num_qubit :
            qc_ansatz.x(2 * pi)
        else :
            qc_ansatz.x(2 * pi - num_qubit + (num_qubit + 1) % 2)
        
    qc_ansatz.compose(qc_pnp, inplace=True)

    return qc_ansatz, pv

#####################################################################################################
### CONST FUNCTION ##################################################################################
#####################################################################################################

def global_cost_function(prob_dist):
    return 1. - prob_dist[0]

def local_cost_function(prob_dist):
    sump = 0.
    num_qubit = int(np.log2(len(prob_dist)))
    for iqubit in range(num_qubit):
        for idx, prob in enumerate(prob_dist):
            if (not ( (idx>>iqubit) & 1 )):
                sump += prob

    return 1. - sump / num_qubit

def general_cost_function(prob_dist, q):
    global_cost = global_cost_function(prob_dist)
    local_cost = local_cost_function(prob_dist)
    return q * global_cost + (1. - q) * local_cost

#####################################################################################################
### CONST FUNCTION SLICE GENERATORS #################################################################
#####################################################################################################

class CostFunctionSliceAllTerms(object):
    def __init__(self):
        self.hop_suppression_threshold = 1.e-4
        self.coeffs = np.zeros(5, dtype='f8')
        self.current = 0.
    
    def fun(self, theta):
        return self.coeffs[0] * np.sin(2. * theta) + \
                self.coeffs[1] * np.cos(2. * theta) + \
                self.coeffs[2] * np.sin(theta) + \
                self.coeffs[3] * np.cos(theta) + \
                self.coeffs[4]
    
    def grad(self, theta):
        return 2. * self.coeffs[0] * np.cos(2. * theta) + \
                (-2. * self.coeffs[1]) * np.sin(2. * theta) + \
                self.coeffs[2] * np.cos(theta) + \
                (-self.coeffs[3]) * np.sin(theta)
    
    def minimize(self):
        if self.grad(0.) < 0.:
            x0 = 0.
            shift = 0.01
        else:
            x0 = 2. * np.pi
            shift = -0.01
            
        fun_array_input = lambda x: self.fun(x[0])

        # first minimum
        res_min1 = sciopt.minimize(fun_array_input, [x0], method='Newton-CG', jac=self.grad)

        negative_fun_array_input = lambda x: -self.fun(x[0])
        negative_grad = lambda x: -self.grad(x)
        
        # first maximum
        res_max1 = sciopt.minimize(negative_fun_array_input, [res_min1.x + shift], method='Newton-CG', jac=negative_grad)
        
        # second minimum
        res_min2 = sciopt.minimize(fun_array_input, [res_max1.x + shift], method='Newton-CG', jac=self.grad)
        
        if (shift > 0. and res_min2.x > 2. * np.pi) or (shift < 0. and res_min2.x < 0.)
            return res_min1.x
        elif abs(res_min1.fun - res_min2.fun) < self.hop_suppression_threshold:
            if abs(res_min1.x - self.current) < abs(res_min2.x - self.current):
                return res_min1.x
            else:
                return res_min2.x
        elif res_min1.fun < res_min2.fun:
            return res_min1.x
        else:
            return res_min2.x
        
class CostFunctionSliceFirst(object):
    def __init__(self):
        self.coeffs = np.zeros(3, dtype='f8')
    
    def fun(self, theta):
        return self.coeffs[0] * np.sin(theta) + \
                self.coeffs[1] * np.cos(theta) + \
                self.coeffs[2]
    
    def grad(self, theta):
        return self.coeffs[0] * np.cos(theta) + \
                (-self.coeffs[1]) * np.sin(theta)
    
    def minimize(self):
        theta_min = np.arctan2(self.coeffs[0], self.coeffs[1]) + np.pi
        if theta_min < 0.:
            theta_min += 2. * np.pi
        elif theta_min > 2. * np.pi:
            theta_min -= 2. * np.pi

        return theta_min

class CostFunctionSliceSecond(object):
    def __init__(self):
        self.coeffs = np.zeros(3, dtype='f8')
        self.current = 0.
    
    def fun(self, theta):
        return self.coeffs[0] * np.sin(2. * theta) + \
                self.coeffs[1] * np.cos(2. * theta) + \
                self.coeffs[2]
    
    def grad(self, theta):
        return 2. * self.coeffs[0] * np.cos(2. * theta) + \
                (-2. * self.coeffs[1]) * np.sin(2. * theta)
    
    def minimize(self):
        theta_min = np.arctan2(self.coeffs[0], self.coeffs[1]) / 2. + np.pi / 2.
        if theta_min < 0.:
            theta_min += 2. * np.pi
        elif theta_min > 2. * np.pi:
            theta_min -= 2. * np.pi
            
        if abs(theta_min - self.current) > np.pi / 2.:
            if theta_min < np.pi:
                theta_min += np.pi
            else:
                theta_min -= np.pi

        return theta_min


class AnalyticAllTerms(CostFunctionSliceAllTerms):
    def set_thetas(self, current):
        self.current = current
        self.thetas = np.array([current, current + np.pi / 4., current - np.pi / 4., current + np.pi / 2., current - np.pi / 2.])
        
    def set_coeffs(self, costs):
        z0, z1, z2, z3, z4 = costs

        self.coeffs[4] = (np.sqrt(2) * (z1 + z2) - z4 - z3 - 2 * z0) / (2 * np.sqrt(2) - 4)
        self.coeffs[2] = (
            1
            / 2.0
            * (
                (z3 - z4) * np.cos(self.current)
                + np.sqrt(2) * (z1 + z2) * np.sin(self.current)
                - 2 * np.sqrt(2) * self.coeffs[4] * np.sin(self.current)
            )
        )
        self.coeffs[3] = (
            -1
            / 2.0
            * (
                (z3 - z4) * np.sin(self.current)
                - np.sqrt(2) * (z1 + z2) * np.cos(self.current)
                + 2 * np.sqrt(2) * self.coeffs[4] * np.cos(self.current)
            )
        )
        self.coeffs[1] = (
            np.cos(2 * self.current) * (self.coeffs[2] * np.cos(self.current) - self.coeffs[3] * np.sin(self.current) + self.coeffs[4])
            + np.sin(2 * self.current)
            * (
                (self.coeffs[2] + self.coeffs[3]) * np.cos(self.current) / np.sqrt(2)
                + (self.coeffs[2] - self.coeffs[3]) * np.sin(self.current) / np.sqrt(2)
                + self.coeffs[4]
            )
            - z3 * np.cos(2 * self.current)
            - z1 * np.sin(2 * self.current)
        )
        self.coeffs[0] = (
            np.sin(2 * self.current) * (self.coeffs[2] * np.cos(self.current) - self.coeffs[3] * np.sin(self.current) + self.coeffs[4])
            - np.cos(2 * self.current)
            * (
                (self.coeffs[2] + self.coeffs[3]) * np.cos(self.current) / np.sqrt(2)
                + (self.coeffs[2] - self.coeffs[3]) * np.sin(self.current) / np.sqrt(2)
                + self.coeffs[4]
            )
            - z3 * np.sin(2 * self.current)
            + z1 * np.cos(2 * self.current)
        )
    
class AnalyticSecond(CostFunctionSliceSecond):
    def set_thetas(self, current):
        self.current = current
        self.thetas = np.array([current, current + np.pi / 4., current - np.pi / 4.])
        
    def set_coeffs(self, costs):
        z0, z1, z2 = costs

        self.coeffs[2] = (z1 + z2) / 2.
        
        c = np.cos(2. * self.current)
        s = np.sin(2. * self.current)
        
        self.coeffs[0] = ((z0 - z2) * (c + s) - (z0 - z1) * (c - s)) / 2.
        self.coeffs[1] = ((z0 - z2) * (c - s) + (z0 - z1) * (c + s)) / 2.
    
class AnalyticFirst(CostFunctionSliceFirst):
    def set_thetas(self, current):
        self.thetas = np.array([current, current + np.pi / 2., current - np.pi / 2.])
        
    def set_coeffs(self, costs):
        z0, z1, z2 = costs

        self.coeffs[2] = (z1 + z2) / 2.
        
        c = np.cos(self.current)
        s = np.sin(self.current)
        
        self.coeffs[0] = ((z0 - z2) * (c + s) - (z0 - z1) * (c - s)) / 2.
        self.coeffs[1] = ((z0 - z2) * (c - s) + (z0 - z1) * (c + s)) / 2.
    
class MatrixCostFunctionSlice(object):
    def set_coeffs(self, costs):
        self.coeffs = self.inverse_matrix @ costs
    
class MatrixAllTerms(CostFunctionSliceAllTerms, MatrixCostFunctionSlice):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 2., current + np.pi / 2., 5, endpoint=True)
        matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        self.current = current
    
class MatrixSecond(CostFunctionSliceSecond, MatrixCostFunctionSlice):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 4., current + np.pi / 4., 3, endpoint=True)
        matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        self.current = current
        
class MatrixFirst(CostFunctionSliceFirst, MatrixCostFunctionSlice):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi / 2., current + np.pi / 2., 3, endpoint=True)
        matrix = np.stack((np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.inverse_matrix = np.linalg.inv(matrix)
        
class FitCostFunctionSlice(object):
    def __init__(self, points_coeff_ratio=4):
        self.ratio = points_coeff_ratio
    
    def set_coeffs(self, costs):
        # Assuming global cost, cost = 1-prob_dist[0]
        # -> binomial distribution

        def negative_likelihood(coeffs):
            x = self.template_matrix @ coeffs
            return -np.prod(scistats.beta.pdf(x, costs + 1., (1. - costs) + 1.))

        b0 = np.linalg.inv(self.template_matrix[::self.ratio, :]) @ costs[::self.ratio]

        res = sciopt.minimize(negative_likelihood, b0)

        self.coeffs = res.x
    
class FitAllTerms(CostFunctionSliceAllTerms, FitCostFunctionSlice):
    def __init__(self, points_coeff_ratio=4):
        CostFunctionSliceAllTerms.__init__(self)
        FitCostFunctionSlice.__init__(self, points_coeff_ratio)
        
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 5 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)
        self.current = current
    
class FitSecond(CostFunctionSliceSecond, FitCostFunctionSlice):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 3 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(2. * self.thetas), np.cos(2. * self.thetas), np.ones_like(self.thetas)), axis=1)
        self.current = current
    
class FitFirst(CostFunctionSliceFirst, FitCostFunctionSlice):
    def set_thetas(self, current):
        self.thetas = np.linspace(current - np.pi, current + np.pi, 3 * self.ratio, endpoint=False)
        self.template_matrix = np.stack((np.sin(self.thetas), np.cos(self.thetas), np.ones_like(self.thetas)), axis=1)

#####################################################################################################
### FISCPNP #########################################################################################
#####################################################################################################

class FISCPNP:
    def __init__(
        self,
        qc_target: QuantumCircuit,
        ansatz: Tuple[QuantumCircuit, ParameterVector],
        q,
        backend,
        physical_qubits=None,
        num_experiments=1,
        shots=8192,
        error_mitigation_filter=None,
        default_slice_gen=MatrixAllTerms,
        seed=12345
    ):
        self.qc_target = qc_target
        self.qc_ansatz, self.param_vec = ansatz
        self.qc_ansatz_inv = self._conjugate_ansatz()
        self.param_val = None
        self.qc_learning = self.qc_target.compose(self.qc_ansatz_inv)

        self.q = q

        self.backend = backend
        self.physical_qubits = physical_qubits
        self.num_experiments = num_experiments
        self.shots = shots
        
        self.error_mitigation_filter = error_mitigation_filter
        
        self.cost_slice_generators = [default_slice_gen for _ in range(len(self.param_vec))]
        
        self.random_gen = np.random.default_rng(seed)
        
        self.statevector_simulator = None
        self.ideal_costs_step = None
        self.ideal_costs_sweep = None
        
    def smo(
        self,
        initial_param_val,
        max_sweeps=10,
        callbacks_step=[],
        callbacks_sweep=[]
    ):
        assert len(initial_param_val) == len(self.param_vec)
        param_val = np.array(initial_param_val)
        sweep_param_val = param_val.copy()

        cost_values = []
        nexp = 0
        iparam = 0
        
        convergence_distance = 1.e-2
        convergence_cost = 1.e-5
        
        current_sweep_mean_cost = 1.
        
        for isweep in range(max_sweeps):
            sweep_mean_cost = 0.
            
            for iparam in range(len(self.param_vec)):
                cost_slice, costs = self._smo_one_iter(sweep_param_val, iparam)
                
                theta_opt = cost_slice.minimize()
                
                for callback in callbacks_step:
                    callback(isweep, iparam, sweep_param_val, theta_opt, costs, cost_slice)

                sweep_param_val[iparam] = theta_opt
                nexp += len(thetas) * self.num_experiments

                cost_value = cost_slice.fun(theta_opt)
                cost_values.append(cost_value)
                
                sweep_mean_cost += cost_value
                
            for callback in callbacks_sweep:
                callback(isweep, param_val, sweep_param_val)
                
            distance = np.max(np.abs(param_val - sweep_param_val))
            
            if distance < convergence_distance:
                break
                
            sweep_mean_cost /= len(self.param_vec)
            
            if abs(sweep_mean_cost - current_sweep_mean_cost) < convergence_cost:
                break
                
            current_sweep_mean_cost = sweep_mean_cost
            param_val = sweep_param_val.copy()

        return param_val, cost_values, nexp
    
    def ideal_cost_step(self, isweep, iparam, sweep_param_val, theta_opt, costs, cost_slice):
        """Convenience function for tracking the ideal cost through callback
        """
        if self.statevector_simulator is None:
            self.statevector_simulator = Aer.get_backend('statevector_simulator')

        if self.ideal_costs_step is None:
            self.ideal_costs_step = []
            
        circuit = transpile(self._make_circuit(sweep_param_val, iparam, theta_opt), backend=self.statevector_simulator)
        prob_dist = np.square(np.abs(self.statevector_simulator.run(circuit).result().data()['statevector']))
        cost = general_cost_function(prob_dist, self.q)
        
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
        cost = general_cost_function(prob_dist, self.q)
        
        self.ideal_costs_sweep.append(cost)

    def _conjugate_ansatz(
        self,
    ) -> QuantumCircuit:

        ansatz = self.qc_ansatz.copy()
        dag = converters.circuit_to_dag(ansatz).reverse_ops()
        for node in dag.op_nodes() :
            if node.name == 'Agate_reverse' :
                dag.substitute_node(
                    node,
                    make_Agate(node.op.params[0], node.op.params[1]).to_instruction()
                )
        return converters.dag_to_circuit(dag)
    
    def _make_circuit(self, param_val, param_id=None, test_value=None):
        if param_id is not None:
            param_val = np.copy(param_val)
            param_val[param_id] = test_value
            
        param_dict = dict(zip(self.param_vec, param_val))
        
        return self.qc_learning.bind_parameters(param_dict)
    
    def _run_circuits(self, circuits):
        if self.backend.name() != 'statevector_simulator':
            for circuit in circuits:
                circuit.measure(circuit.qregs[0], circuit.cregs[0])
        
        if self.backend.configuration().simulator:
            circuits = transpile(circuits, backend=self.backend, initial_layout=self.physical_qubits, optimization_level=1)
        else:
            circuits = transpile_with_dynamical_decoupling(circuits, backend=self.backend, initial_layout=self.physical_qubits, optimization_level=1)
            
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

            probs = [np.zeros(2 ** self.qc_target.num_qubits, dtype='f8') for _ in range(num_circuits)]
            for ic, counts in enumerate(counts_list):
                total = sum(counts.values())
                for key, value in counts.items():
                    probs[ic][int(key, 2)] = value / total
                
        return probs
        
    def _compute_costs(self, circuits, num_toys=None):
        prob_dists = self._run_circuits(circuits)
        
        costs = np.empty(len(prob_dists), dtype='f8')
        for iprob, prob_dist in enumerate(prob_dists):
            costs[iprob] = general_cost_function(prob_dist, self.q)
            
        if not num_toys:
            return costs
        
        shots = self.shots * self.num_experiments
        sigmas = np.empty_like(costs)
        
        for iprob, prob_dist in enumerate(prob_dists):
            sumw = 0.
            sumw2 = 0.
            for _ in range(num_toys):
                toy_prob_dist = self.random_gen.multinomial(shots, prob_dist) / shots
                toy_cost = general_cost_function(toy_prob_dist, self.q)
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
        
        cost_slice = self.cost_slice_generators[param_id]()
        cost_slice.set_thetas(current)
        
        circuits = []
        for theta in cost_slice.thetas:
            circuits.append(self._make_circuit(param_val, param_id, theta))
        
        costs = self._compute_costs(circuits)
        
        cost_slice.set_coeffs(costs)

        return cost_slice, costs