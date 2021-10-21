from typing import Tuple
import numpy as np
import scipy.optimize as sciopt
from qiskit import QuantumCircuit, QuantumRegister, converters, transpile
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
    return -prob_dist[0]

def local_cost_function(prob_dist):
    local_cost_function = 0.
    num_qubit = int(np.log2(len(prob_dist)))
    for iqubit in range(num_qubit):
        for idx, prob in enumerate(prob_dist):
            if (not ( (idx>>iqubit) & 1 )):
                local_cost_function += prob
    local_cost_function /= num_qubit
    local_cost_function *= -1.

    return local_cost_function

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
        error_mitigation_filter=None
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
        
    def smo(
        self,
        initial_param_val,
        maxfuncall=100,
        reset_interval=1,
    ):
        assert len(initial_param_val) == len(self.param_vec)
        param_val = np.array(initial_param_val)

        loss_values = []
        loss_value = None
        nfuncall = 0
        iparam = 0
        niter = 0

        while nfuncall < maxfuncall:
            niter += 1
            if niter % reset_interval == 0:
                loss_value = None

            theta, ncall, loss_value = self._smo_one_iter(param_val, iparam, loss_value)
            print('niter', niter, 'iparam', iparam, 'loss', loss_value)

            param_val[iparam] = theta
            nfuncall += ncall
            loss_values.append(loss_value)

            iparam = (iparam + 1) % len(self.param_vec)

        return param_val, nfuncall, loss_values[1:]

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
    
    def _make_circuit(self, param_val, param_id, shift):
        local_params = np.copy(param_val)
        local_params[param_id] += shift
        param_dict = dict(zip(self.param_vec, local_params))
        
        return self.qc_learning.bind_parameters(param_dict)
    
    def _get_probabilities(self, raw_counts):
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
    
    def _compute_costs(self, circuits):
        for circuit in circuits:
            circuit.measure(circuit.qregs[0], circuit.cregs[0])
        
        if self.backend.configuration().simulator:
            circuits = transpile(circuits, backend=self.backend, initial_layout=self.physical_qubits, optimization_level=1)
        else:
            circuits = transpile_with_dynamical_decoupling(circuits, backend=self.backend, initial_layout=self.physical_qubits, optimization_level=1)

        circuits_to_run = []
        for circuit in circuits:
            circuits_to_run += [circuit] * self.num_experiments
            
        counts = self.backend.run(circuits_to_run, shots=self.shots).result().get_counts()
        
        prob_dists = self._get_probabilities(counts)

        costs = []
        for prob_dist in prob_dists:
            global_cost = global_cost_function(prob_dist)
            local_cost = local_cost_function(prob_dist)
            costs.append(self.q * global_cost + (1-self.q) * local_cost)
            
        return costs
        
    def _smo_one_iter(
        self,
        param_val,
        param_id,
        z0
    ):
        p_ = param_val[param_id]
        
        nfuncall = 0
        circuits = []

        if z0 is None:
            circuits.append(self._make_circuit(param_val, param_id, 0.))

        circuits.append(self._make_circuit(param_val, param_id, np.pi / 4.))
        circuits.append(self._make_circuit(param_val, param_id, -np.pi / 4.))
        circuits.append(self._make_circuit(param_val, param_id, np.pi / 2.))
        circuits.append(self._make_circuit(param_val, param_id, -np.pi / 2.))
        
        costs = self._compute_costs(circuits)
        nfuncall += len(circuits)
        
        if z0 is None:
            z0, z1, z2, z3, z4 = costs
        else:
            z1, z2, z3, z4 = costs

        b = np.zeros(5)
        b[4] = (np.sqrt(2) * (z1 + z2) - z4 - z3 - 2 * z0) / (2 * np.sqrt(2) - 4)
        b[2] = (
            1
            / 2.0
            * (
                (z3 - z4) * np.cos(p_)
                + np.sqrt(2) * (z1 + z2) * np.sin(p_)
                - 2 * np.sqrt(2) * b[4] * np.sin(p_)
            )
        )
        b[3] = (
            -1
            / 2.0
            * (
                (z3 - z4) * np.sin(p_)
                - np.sqrt(2) * (z1 + z2) * np.cos(p_)
                + 2 * np.sqrt(2) * b[4] * np.cos(p_)
            )
        )
        b[1] = (
            np.cos(2 * p_) * (b[2] * np.cos(p_) - b[3] * np.sin(p_) + b[4])
            + np.sin(2 * p_)
            * (
                (b[2] + b[3]) * np.cos(p_) / np.sqrt(2)
                + (b[2] - b[3]) * np.sin(p_) / np.sqrt(2)
                + b[4]
            )
            - z3 * np.cos(2 * p_)
            - z1 * np.sin(2 * p_)
        )
        b[0] = (
            np.sin(2 * p_) * (b[2] * np.cos(p_) - b[3] * np.sin(p_) + b[4])
            - np.cos(2 * p_)
            * (
                (b[2] + b[3]) * np.cos(p_) / np.sqrt(2)
                + (b[2] - b[3]) * np.sin(p_) / np.sqrt(2)
                + b[4]
            )
            - z3 * np.sin(2 * p_)
            + z1 * np.cos(2 * p_)
        )

        # calculate minimum point of f
        def f(theta, b):
            return (
                b[0] * np.sin(2 * theta)
                + b[1] * np.cos(2 * theta)
                + b[2] * np.sin(theta)
                + b[3] * np.cos(theta)
                + b[4]
            )

        l = 0
        r = 2 * np.pi
        theta = 0
        cost = 1.0

        eps = 1e-16
        Ns = 10000

        while r - l > eps:
            result = sciopt.brute(f, ranges=((l, r),), args=(b,), full_output=True, Ns=Ns)
            theta = result[0][0]
            cost = result[1]
            l, r = theta - (r - l) / Ns, theta + (r - l) / Ns

        return theta, nfuncall, z0
