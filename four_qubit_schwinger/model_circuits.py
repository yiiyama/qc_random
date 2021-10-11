import numpy as np
import collections
from qiskit import QuantumCircuit, pulse, transpile
from qiskit import schedule as build_schedule
from qiskit.circuit import Gate, Parameter

from calibrations import LinearizedCR, cx_circuit

nsites = 4

def single_step_no_opt(J, mu, omegadt):
    """Naive single-step circuit for reference.
    """
    
    circuit = QuantumCircuit(nsites)
    
    for j in range(nsites - 1):
        circuit.rxx(omegadt, j, j + 1)
        circuit.ryy(omegadt, j, j + 1)

        for k in range(j):
            circuit.rzz(J * (nsites - j - 1) * omegadt, k, j)
            
    for j in range(nsites):
        angle = (mu * (-1. if j % 2 == 0 else 1.) - J * ((nsites - j) // 2)) * omegadt
        circuit.rz(angle, j)
        
    return circuit

def _single_step_no_last_rzz_swap(J, mu, omegadt, crs=None, cxs=None):
    circuit = QuantumCircuit(nsites)
    
    # First step
    
    for j in range(nsites):
        angle = (mu * (-1. if j % 2 == 0 else 1.) - J * ((nsites - j) // 2)) * omegadt
        circuit.rz(angle, j)

    if crs is None:
        circuit.rxx(omegadt, 0, 1)
        circuit.ryy(omegadt, 0, 1)
        circuit.rxx(omegadt, 2, 3)
        circuit.ryy(omegadt, 2, 3)
        circuit.rxx(omegadt, 1, 2)
        circuit.ryy(omegadt, 1, 2)
        circuit.rzz(omegadt, 1, 2)
    else:
        circuit.compose(crs[(0, 1)].rxx_circuit(omegadt), qubits=(0, 1), inplace=True)
        circuit.compose(crs[(0, 1)].ryy_circuit(omegadt), qubits=(0, 1), inplace=True)
        circuit.compose(crs[(2, 3)].rxx_circuit(omegadt), qubits=(2, 3), inplace=True)
        circuit.compose(crs[(2, 3)].ryy_circuit(omegadt), qubits=(2, 3), inplace=True)
        circuit.compose(crs[(1, 2)].rxx_circuit(omegadt), qubits=(1, 2), inplace=True)
        circuit.compose(crs[(1, 2)].ryy_circuit(omegadt), qubits=(1, 2), inplace=True)
        circuit.compose(crs[(1, 2)].rzz_circuit(omegadt), qubits=(1, 2), inplace=True)
    
    ## rzz(2 * omegadt, 0, 1)
    if cxs is None:
        circuit.cx(0, 1)
    else:
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)
    circuit.rz(2 * omegadt, 1)
    #circuit.cx(0, 1)

    ## rzz(omegadt, 0, 2)
    #circuit.cx(0, 1)
    if cxs is None:
        circuit.cx(1, 0)
        circuit.cx(0, 1)
    else:
        circuit.compose(cxs[(1, 0)], qubits=(1, 0), inplace=True)
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)
        
    return circuit

def single_step(J, mu, omegadt, crs=None, cxs=None):
    circuit = _single_step_no_last_rzz_swap(J, mu, omegadt, crs=crs, cxs=cxs)
    if crs is None:
        circuit.rzz(omegadt, 1, 2)
    else:
        circuit.compose(crs[(1, 2)].rzz_circuit(omegadt), qubits=(1, 2), inplace=True)
        
    if cxs is None:
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
    else:
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)
        circuit.compose(cxs[(1, 0)], qubits=(1, 0), inplace=True)
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)
    
    return circuit

def two_steps(J, mu, omegadt, crs=None, cxs=None):
    # First step
    
    circuit = _single_step_no_last_rzz_swap(J, mu, omegadt, crs=crs, cxs=cxs)
    #circuit.rzz(omegadt, 1, 2)
    #circuit.cx(0, 1)
    #circuit.cx(1, 0)
    #circuit.cx(0, 1)

    if crs is None:
        circuit.rzz(2 * omegadt, 1, 2)
    else:
        circuit.compose(crs[(1, 2)].rzz_circuit(2 * omegadt), qubits=(1, 2), inplace=True)
    
    # Second step

    # rzz(omegadt, 0, 2)
    #circuit.cx(0, 1)
    #circuit.cx(1, 0)
    #circuit.cx(0, 1)
    #circuit.rzz(omegadt, 1, 2)
    if cxs is None:
        circuit.cx(0, 1)
        circuit.cx(1, 0)
    else:
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)
        circuit.compose(cxs[(1, 0)], qubits=(1, 0), inplace=True)
    #circuit.cx(0, 1)
    
    # rzz(2 * omegadt, 0, 1)
    #circuit.cx(0, 1)
    circuit.rz(2 * omegadt, 1)
    if cxs is None:
        circuit.cx(0, 1)
    else:
        circuit.compose(cxs[(0, 1)], qubits=(0, 1), inplace=True)

    if crs is None:
        circuit.rxx(omegadt, 1, 2)
        circuit.ryy(omegadt, 1, 2)
        circuit.rzz(omegadt, 1, 2)
        circuit.rxx(omegadt, 0, 1)
        circuit.ryy(omegadt, 0, 1)
        circuit.rxx(omegadt, 2, 3)
        circuit.ryy(omegadt, 2, 3)
    else:
        circuit.compose(crs[(1, 2)].rxx_circuit(omegadt), qubits=(1, 2), inplace=True)
        circuit.compose(crs[(1, 2)].ryy_circuit(omegadt), qubits=(1, 2), inplace=True)
        circuit.compose(crs[(1, 2)].rzz_circuit(omegadt), qubits=(1, 2), inplace=True)
        circuit.compose(crs[(0, 1)].rxx_circuit(omegadt), qubits=(0, 1), inplace=True)
        circuit.compose(crs[(0, 1)].ryy_circuit(omegadt), qubits=(0, 1), inplace=True)
        circuit.compose(crs[(2, 3)].rxx_circuit(omegadt), qubits=(2, 3), inplace=True)
        circuit.compose(crs[(2, 3)].ryy_circuit(omegadt), qubits=(2, 3), inplace=True)
       
    for j in range(nsites):
        angle = (mu * (-1. if j % 2 == 0 else 1.) - J * ((nsites - j) // 2)) * omegadt
        circuit.rz(angle, j)
        
    return circuit

def add_dynamical_decoupling(circuit, backend):
    sched = build_schedule(circuit, backend=backend)
    
    channels_config = backend.configuration().channels
    
    channel_qubit_map = {}
    for channel in sched.channels:
        channel_config = backend.configuration().channels[channel.name]
        channel_qubit_map[channel.name] = channel_config['operates']['qubits'][0]
        
    busy_timeslots = collections.defaultdict(list)
    for tstart, inst in sched.instructions:
        busy_timeslots[channel_qubit_map[inst.channel.name]].append((tstart, tstart + inst.duration))
        
    calibrations = backend.defaults().instruction_schedule_map
    for qubit in set(channel_qubit_map.values()):
        x_inst = calibrations.get('x', [qubit]).instructions[0][1]
        x_duration = x_inst.duration
            
        interval_start = 0
        for tstart, tend in sorted(busy_timeslots[qubit]):
            idle_time = tstart - interval_start
            num_insertable = idle_time // (2 * x_duration)
            if num_insertable == 0:
                interval_start = tend
                continue
                
            with pulse.build(backend=backend) as dd_sched:
                with pulse.align_equispaced(duration=idle_time):
                    for _ in range(2 * num_insertable):
                        pulse.play(x_inst.pulse, x_inst.channel)
                    
            dd_sched = pulse.transforms.block_to_schedule(dd_sched)
            
            sched.insert(interval_start, dd_sched, inplace=True)
            
            interval_start = tend
            
    return sched