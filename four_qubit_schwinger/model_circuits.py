from qiskit import QuantumCircuit

def single_step_no_opt(num_site, J, mu, omegadt):
    """Naive single-step circuit for reference.
    """
    
    circuit = QuantumCircuit(num_site)
    
    for j in range(0, num_site - 1, 2):
        circuit.rxx(omegadt, j, j + 1)
        circuit.ryy(omegadt, j, j + 1)

    for j in range(1, num_site - 1, 2):
        circuit.rxx(omegadt, j, j + 1)
        circuit.ryy(omegadt, j, j + 1)

    for j in range(num_site - 1):
        for k in range(j):
            circuit.rzz(J * (num_site - j - 1) * omegadt, k, j)    

    for j in range(num_site):
        angle = (mu * (-1. if j % 2 == 0 else 1.) - J * ((num_site - j) // 2)) * omegadt
        circuit.rz(angle, j)
        
    return circuit

def _rxxryy(num_site, circuit, omegadt, qubits, rtts=None):
    if rtts is None:
        #circuit.rxx(omegadt, *qubits)
        #circuit.ryy(omegadt, *qubits)
        circuit.h(qubits)
        circuit.s(qubits)
        circuit.cx(*qubits)
        circuit.rx(omegadt, qubits[0])
        circuit.rz(omegadt, qubits[1])
        circuit.cx(*qubits)
        circuit.sdg(qubits)
        circuit.h(qubits)
    else:
        circuit.compose(rtts[qubits].rxxryy_circuit(omegadt), qubits=qubits, inplace=True)

def _single_step_no_last_rzz_and_swap(num_site, J, mu, omegadt, rtts=None, cxs=None):
    circuit = QuantumCircuit(num_site)
    
    for j in range(num_site):
        angle = (mu * (-1. if j % 2 == 0 else 1.) - J * ((num_site - j) // 2)) * omegadt
        circuit.rz(angle, j)

    for j in range(0, num_site - 1, 2):
        _rxxryy(num_site, circuit, omegadt, (j, j + 1), rtts=rtts)

    for j in range(1, num_site - 1, 2):
        _rxxryy(num_site, circuit, omegadt, (j, j + 1), rtts=rtts)

    if num_site == 4:
        k = 1
        j = 2
        angle = J * (num_site - j - 1) * omegadt
        if rtts is None:
            circuit.rzz(angle, k, j)
        else:
            circuit.compose(rtts[(k, j)].rzz_circuit(angle), qubits=(1, 2), inplace=True)

        k = 0
        j = 1
        angle = J * (num_site - j - 1) * omegadt
        ## rzz(angle, k, j)
        if cxs is None:
            circuit.cx(k, j)
        else:
            circuit.compose(cxs[(k, j)], qubits=(k, j), inplace=True)
        circuit.rz(angle, j)
        circuit.cx(k, j) # cancels with swap below

        k = 0
        ## swap(k, k + 1)
        circuit.cx(k, k + 1) # cancels with rzz above
        if cxs is None:
            circuit.cx(k + 1, k)
            circuit.cx(k, k + 1)
        else:
            circuit.compose(cxs[(k + 1, k)], qubits=(k + 1, k), inplace=True)
            circuit.compose(cxs[(k, k + 1)], qubits=(k, k + 1), inplace=True)

        ## ~~rzz(angle, k + 1, j)~~
        ## ~~swap(k, k + 1)~~
        
    return circuit

def _single_step_no_first_swap_and_rzz(num_site, J, mu, omegadt, rtts=None, cxs=None):
    circuit = QuantumCircuit(num_site)
    
    ## ~~swap(k, k + 1)~~
    ## ~~rzz(angle, k + 1, j)~~
    
    for j in range(num_site):
        if j in (0, 1):
            jj = 1 - j
        else:
            jj = j
        angle = (mu * (-1. if jj % 2 == 0 else 1.) - J * ((num_site - jj) // 2)) * omegadt
        circuit.rz(angle, j)

    if num_site == 4:    
        k = 0
        ## swap(k, k + 1)
        if cxs is None:
            circuit.cx(k, k + 1)
            circuit.cx(k + 1, k)
        else:
            circuit.compose(cxs[(k, k + 1)], qubits=(k, k + 1), inplace=True)
            circuit.compose(cxs[(k + 1, k)], qubits=(k + 1, k), inplace=True)
        circuit.cx(k, k + 1) # cancels with rzz below

        k = 0
        j = 1
        angle = J * (num_site - j - 1) * omegadt
        ## rzz(angle, k, j)
        circuit.cx(k, j) # cancels with swap above
        circuit.rz(angle, j)
        if cxs is None:
            circuit.cx(k, j)
        else:
            circuit.compose(cxs[(k, j)], qubits=(k, j), inplace=True)

        k = 1
        j = 2
        angle = J * (num_site - j - 1) * omegadt
        if rtts is None:
            circuit.rzz(angle, k, j)
        else:
            circuit.compose(rtts[(k, j)].rzz_circuit(angle), qubits=(1, 2), inplace=True)

    for j in range(1, num_site - 1, 2):
        _rxxryy(num_site, circuit, omegadt, (j, j + 1), rtts=rtts)
        
    for j in range(0, num_site - 1, 2):
        _rxxryy(num_site, circuit, omegadt, (j, j + 1), rtts=rtts)
    
    return circuit

def single_step(num_site, J, mu, omegadt, rtts=None, cxs=None):
    circuit = _single_step_no_last_rzz_and_swap(num_site, J, mu, omegadt, rtts=rtts, cxs=cxs)

    if num_site == 4:
        k = 0
        j = 2
        angle = J * (num_site - j - 1) * omegadt
        ## rzz(angle, k + 1, j)
        if rtts is None:
            circuit.rzz(angle, k + 1, j)
        else:
            circuit.compose(rtts[(k + 1, j)].rzz_circuit(angle), qubits=(k + 1, j), inplace=True)

        ## swap(k, k + 1)
        if cxs is None:
            circuit.swap(k, k + 1)
        else:
            circuit.compose(cxs[(k, k + 1)], qubits=(k, k + 1), inplace=True)
            circuit.compose(cxs[(k + 1, k)], qubits=(k + 1, k), inplace=True)
            circuit.compose(cxs[(k, k + 1)], qubits=(k, k + 1), inplace=True)
    
    return circuit

def two_steps(num_site, J, mu, omegadt, rtts=None, cxs=None):
    # Begin first step
    
    circuit = _single_step_no_last_rzz_and_swap(num_site, J, mu, omegadt, rtts=rtts, cxs=cxs)
    
    if num_site == 4:
        k = 0
        j = 2
        angle = J * (num_site - j - 1) * omegadt
        ## rzz(angle, k + 1, j) -> combine with rzz below

        ## swap(k, k + 1) -> cancel with the swap below
    
    # End first step

        if rtts is None:
            circuit.rzz(2. * angle, k + 1, j)
        else:
            circuit.compose(rtts[(k + 1, j)].rzz_circuit(2. * angle), qubits=(k + 1, j), inplace=True)
    
    # Begin second step

    ## swap(k, k + 1) -> cancel with the swap above
    
    ## rzz(angle, k + 1, j) -> combine with rzz above
    
    circuit.compose(_single_step_no_first_swap_and_rzz(num_site, J, mu, omegadt, rtts=rtts, cxs=cxs), inplace=True)
    
    # End second step
        
    return circuit
