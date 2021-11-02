import os
import sys
import numpy as np
import logging
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.ignis.mitigation.measurement import complete_meas_cal, MeasurementFilter
from qiskit.result.counts import Counts

logging.basicConfig(level=logging.WARNING, format='%(asctime)s: %(message)s')
logging.getLogger('four_qubit_schwinger')
logger = logging.getLogger('four_qubit_schwinger.main')

sys.path.append(os.path.dirname(__file__) + '/..')

## include ../ ##
from rttgen import CNOTBasedRtt, PulseEfficientCR

## include ../ ##
from cx_decomposition import cx_circuit

## include ../ ##
from model_circuits import single_step, two_steps

## include ../ ##
from trotter import trotter_step_circuits

## include ../ ##
from transpile_with_dd import transpile_with_dynamical_decoupling

## include ../ ##
from pnp_ansatze import make_pnp_ansatz

## include ../ ##
from cost_functions import global_cost_function, local_cost_function

## include ../ ##
from cost_sections import FitSecond, FitFirst, FitGeneral, FitSymmetric

## include ../ ##
from sequential_minimizer import SequentialVCMinimizer, SectionGenSwitcher

#####################################################################################################
### FORWARD CIRCUIT COMPONENTS ######################################################################
#####################################################################################################

def make_step_circuits(num_sites, aJ, am, omegadt, backend, physical_qubits):
    qubit_pairs = list(zip(range(0, num_sites - 1), range(1, num_sites)))
  
    if backend.configuration().simulator:
        rtts = dict((qubits, CNOTBasedRtt(backend, qubits)) for qubits in qubit_pairs)
        cxs = dict((qubits, cx_circuit(backend, *qubits)) for qubits in qubit_pairs)
        cxs.update((qubits[::-1], cx_circuit(backend, *qubits[::-1])) for qubits in qubit_pairs)
    else:
        rtts = dict((qubits, PulseEfficientCR(backend, (physical_qubits[qubits[0]], physical_qubits[qubits[1]]))) for qubits in qubit_pairs)
        cxs = dict((qubits, cx_circuit(backend, physical_qubits[qubits[0]], physical_qubits[qubits[1]])) for qubits in qubit_pairs)
        cxs.update((qubits[::-1], cx_circuit(backend, physical_qubits[qubits[1]], physical_qubits[qubits[0]])) for qubits in qubit_pairs)

    single_step_circuit = single_step(num_sites, aJ, am, omegadt, rtts=rtts, cxs=cxs)
    two_step_circuit = two_steps(num_sites, aJ, am, omegadt, rtts=rtts, cxs=cxs)
    
    return single_step_circuit, two_step_circuit

#####################################################################################################
### FORWARD STEPS ###################################################################################
#####################################################################################################

def combine_counts(new, base):
    data = dict(base)
    for key, value in new.items():
        try:
            data[key] += value
        except KeyError:
            data[key] = value
                
    return Counts(data, time_taken=base.time_taken, creg_sizes=base.creg_sizes, memory_slots=base.memory_slots)

def run_forward_circuits(
    target_circuits,
    backend,
    initial_layout=None,
    shots=8192,
    error_mitigation_filter=None):
    
    circuits = []
    for target_circuit in target_circuits:
        circuit = target_circuit.measure_all(inplace=False)
        circuits.append(circuit)
    
    if backend.configuration().simulator:
        transpile_fn = transpile
    else:
        transpile_fn = transpile_with_dynamical_decoupling
        
    circuits = transpile_fn(circuits, backend=backend, initial_layout=initial_layout, optimization_level=1)

    max_shots = backend.configuration().max_shots
    if shots > max_shots:
        circuits *= shots // max_shots
        shots = max_shots

    logger.info('Running {} circuits, {} shots per experiment, {} experiments'.format(len(target_circuits), shots, len(circuits)))
        
    job = backend.run(circuits, shots=shots)
    counts_list_tmp = job.result().get_counts()
    
    logger.info('Forward circuit results returned')
    
    counts_list = counts_list_tmp[:len(target_circuits)]
    for it, counts in enumerate(counts_list_tmp[len(target_circuits):]):
        ic = it % len(target_circuits)
        counts_list[ic] = combine_counts(counts, counts_list[ic])
        
    if error_mitigation_filter is not None:
        for ic, counts in enumerate(counts_list):
            counts_list[ic] = error_mitigation_filter.apply(counts)

    return counts_list

#####################################################################################################
### RQD STEP ########################################################################################
#####################################################################################################

def rqd_step(
    num_tsteps,
    forward_circuits,
    backend,
    physical_qubits,
    max_sweeps,
    minimizer_shots,
    forward_shots,
    error_mitigation_filter,
    optimal_params):
    
    num_sites = forward_circuits[0].num_qubits
    approximator = make_pnp_ansatz(
        num_qubits=num_sites,
        num_layers=num_sites // 2,
        initial_x_positions=[1, 2],
        structure=[(1, 2), (0, 1), (2, 3)],
        first_layer_structure=[(0, 1), (2, 3)])
    
    if optimal_params is None:
        initial_state = None
    else:
        initial_state = approximator.bind_parameters(dict(zip(approximator.parameters, optimal_params)))

    target_circuits = trotter_step_circuits(num_tsteps, forward_circuits, initial_state=initial_state, measure=False)

    forward_counts = run_forward_circuits(
        target_circuits,
        backend,
        initial_layout=physical_qubits,
        shots=forward_shots,
        error_mitigation_filter=error_mitigation_filter)

    compiler_circuit = target_circuits[-1].compose(approximator.inverse(), inplace=False)

    minimizer = SequentialVCMinimizer(
        compiler_circuit,
        local_cost_function,
        backend,
        strategy='largest-drop',
        error_mitigation_filter=error_mitigation_filter,
        shots=minimizer_shots,
        default_section=FitGeneral)
    
    minimizer.cost_section_generators[0] = FitSecond
    minimizer.cost_section_generators[1] = FitFirst
    minimizer.cost_section_generators[2] = FitSecond
    minimizer.cost_section_generators[3] = FitFirst
    minimizer.cost_section_generators[5] = FitFirst
    
    minimizer.callbacks_sweep.append(SequentialVCMinimizer.ideal_cost_sweep)

    switcher = SectionGenSwitcher(FitSymmetric, [4, 6, 7, 8, 9], 0.015)
    minimizer.callbacks_sweep.append(switcher.callback_sweep)

    initial_params = np.ones(len(approximator.parameters)) * np.pi / 4.
    param_val, total_shots = minimizer.minimize(initial_params, max_sweeps=max_sweeps)

    shots_values = np.array(minimizer.shots_sweep)
    cost_values = np.array(minimizer.ideal_costs_sweep)
    
    return forward_counts, param_val, total_shots, shots_values, cost_values

#####################################################################################################
### MAIN ############################################################################################
#####################################################################################################

def main(backend, user_messenger, **kwargs):
    """Main entry point of the program.

    Args:
        backend: Backend to submit the circuits to.
        user_messenger: Used to communicate with the program consumer.
        kwargs: User inputs.
    """
    
    num_sites = kwargs['num_sites']
    aJ = kwargs['aJ']
    am = kwargs['am']
    omegadt = kwargs['omegadt']
    num_tsteps = kwargs['num_tsteps']
    tsteps_per_rqd = kwargs['tsteps_per_rqd']
    physical_qubits = kwargs.get('physical_qubits', None)
    error_matrix = kwargs.get('error_matrix', None)
    max_sweeps = kwargs.get('max_sweeps', 100)
    minimizer_shots = kwargs.get('minimizer_shots', 4096)
    forward_shots = kwargs.get('foward_shots', 2 * 8192)
    
    if error_matrix is not None:
        _, state_labels = complete_meas_cal(qubit_list=list(range(num_sites)), qr=QuantumRegister(num_sites), circlabel='mcal')
        error_mitigation_filter = MeasurementFilter(error_matrix, state_labels)
    else:
        error_mitigation_filter = None
        
    forward_circuits = make_step_circuits(num_sites, aJ, am, omegadt, backend, physical_qubits)
    
    forward_counts_list = []
    optimal_params_list = []

    optimal_params = None

    for it in range(num_tsteps // tsteps_per_rqd):
        logger.info('Starting RQD step {}'.format(it))
        
        forward_counts, optimal_params, total_shots, shots_values, cost_values = rqd_step(
            tsteps_per_rqd,
            forward_circuits,
            backend,
            physical_qubits,
            max_sweeps,
            minimizer_shots, 
            forward_shots,
            error_mitigation_filter,
            optimal_params)
        
        logger.info('Completed RQD step {}'.format(it))

        user_messenger.publish({"rqd_step": it, "total_shots": total_shots, "shots_values": shots_values, "cost_values": cost_values})

        forward_counts_list.append(forward_counts)
        optimal_params_list.append(optimal_params)

    return forward_counts_list, optimal_params_list
