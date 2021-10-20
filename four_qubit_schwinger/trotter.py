from qiskit import QuantumCircuit, transpile

from transpile_with_dd import transpile_with_dynamical_decoupling

def trotter_step_circuits(num_steps, single_step_circuit, two_step_circuit=None, backend=None, physical_qubits=None, optimization_level=1, with_dd=False, circuit_multiplicity=1):
    nsites = single_step_circuit.num_qubits
    
    circuits = []
    for nrep in range(1, num_steps + 1):
        circuit = QuantumCircuit(nsites, nsites)
        circuit.x(range(0, nsites, 2))

        if two_step_circuit is None:
            for _ in range(nrep):
                circuit.compose(single_step_circuit, inplace=True)
        else:
            for _ in range(nrep // 2):
                circuit.compose(two_step_circuit, inplace=True)

            if nrep % 2 == 1:
                circuit.compose(single_step_circuit, inplace=True)
            
        circuit.measure(circuit.qregs[0], circuit.cregs[0])
        circuits.append(circuit)
        
    if with_dd:
        circuits = transpile_with_dynamical_decoupling(circuits, backend=backend, initial_layout=physical_qubits, optimization_level=optimization_level)
    else:
        circuits = transpile(circuits, backend=backend, initial_layout=physical_qubits, optimization_level=optimization_level)
    circuits *= circuit_multiplicity
    
    return circuits
