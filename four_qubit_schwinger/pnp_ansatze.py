from qiskit import QuantumCircuit

def make_Agate(theta, phi):
    circuit = QuantumCircuit(2, name='A')

    circuit.cx(1, 0)
    circuit.rz(-phi, 1)
    circuit.ry(-theta, 1)
    circuit.cx(0, 1)
    circuit.ry(theta, 1)
    circuit.rz(phi, 1)
    circuit.cx(1, 0)

    return circuit.to_instruction()

def make_pnp_ansatz(num_qubits, num_layers, initial_state, add_barriers=False):
    circuit  = QuantumCircuit(num_qubits)

    circuit.x(initial_state)
    
    if add_barriers:
        circuit.barrier()
    
    num_parameters = (num_qubits - 1) * num_layers * 2
    pv = ParameterVector('\N{greek small letter theta}', num_parameters)
    iparam = 0
        
    for ilayer in range(num_layers):
        for iqubit in range(0, num_qubits - 1, 2):
            Agate = make_Agate(*pv[iparam:iparam + 2])
            iparam += 2
            circuit.append(Agate, (iqubit, iqubit + 1))

        for iqubit in range(1, num_qubits - 1, 2):
            Agate = make_Agate(*pv[iparam:iparam + 2])
            iparam += 2
            circuit.append(Agate, (iqubit, iqubit + 1))

        if add_barriers:
            circuit.barrier()

    return circuit
