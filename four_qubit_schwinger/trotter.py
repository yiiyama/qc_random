from qiskit import QuantumCircuit, transpile

def trotter_step_circuits(num_steps, single_step_circuit, two_step_circuit=None, initial_state=None, measure=True):
    nsites = single_step_circuit.num_qubits
    
    circuits = []
    for nrep in range(1, num_steps + 1):
        circuit = QuantumCircuit(nsites, nsites)
        if initial_state is None:
            circuit.x(range(0, nsites, 2))
        else:
            circuit.compose(initial_state, inplace=True)

        if two_step_circuit is None:
            for _ in range(nrep):
                circuit.compose(single_step_circuit, inplace=True)
        else:
            for _ in range(nrep // 2):
                circuit.compose(two_step_circuit, inplace=True)

            if nrep % 2 == 1:
                circuit.compose(single_step_circuit, inplace=True)

        if measure:    
            circuit.measure(circuit.qregs[0], circuit.cregs[0])
            
        circuits.append(circuit)
    
    return circuits
