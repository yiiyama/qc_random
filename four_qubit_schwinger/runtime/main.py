import os
import sys
import re
import numpy as np
import logging
from qiskit import QuantumCircuit, QuantumRegister, transpile, IBMQ, Aer
from qiskit.converters import circuit_to_dag
from qiskit.ignis.mitigation.measurement import complete_meas_cal, MeasurementFilter
from qiskit.result.counts import Counts
from qiskit.providers.ibmq import IBMQAccountError

logging.basicConfig(level=logging.WARNING, format='%(asctime)s: %(message)s')
logging.getLogger('schwinger_rqd')
logger = logging.getLogger('schwinger_rqd.main')

## ignore
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

## include ../
from rttgen import CNOTBasedRtt, PulseEfficientCR

## include ../
from cx_decomposition import cx_circuit

## include ../
from model_circuits import single_step, two_steps

## include ../
from trotter import trotter_step_circuits

## include ../
from transpile_with_dd import transpile_with_dynamical_decoupling

## include ../
from pnp_ansatze import make_pnp_ansatz

## include ../
from cost_functions import global_cost_function, local_cost_function

## include ../
from cost_sections import FitSecond, FitFirst, FitGeneral, FitSymmetric

## include ../
from sequential_minimizer import SequentialVCMinimizer, SectionGenSwitcher, IdealCost

#####################################################################################################
### FORWARD CIRCUIT COMPONENTS ######################################################################
#####################################################################################################

def make_step_circuits(num_sites, aJ, am, omegadt, backend, physical_qubits=None):
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
### FISC STEP ########################################################################################
#####################################################################################################

def run_fisc(
    compiler_circuit,
    backend,
    physical_qubits,
    max_sweeps,
    shots,
    error_mitigation_filter,
    ideal_cost=None,
    publish_sweep_result=None,
    resume_from=None):

    section_generators = [FitGeneral] * len(compiler_circuit.parameters)
    if compiler_circuit.num_qubits == 2:
        section_generators[0] = FitSecond
        section_generators[1] = FitFirst
    elif compiler_circuit.num_qubits == 4:
        section_generators[0] = FitSecond
        section_generators[1] = FitFirst
        section_generators[2] = FitSecond
        section_generators[3] = FitFirst
        section_generators[5] = FitFirst
        
    if backend.configuration().simulator:
        transpile_fn = transpile
    else:
        transpile_fn = transpile_with_dynamical_decoupling    

    minimizer = SequentialVCMinimizer(
        compiler_circuit,
        local_cost_function,
        section_generators,
        backend,
        strategy='largest-drop',
        error_mitigation_filter=error_mitigation_filter,
        transpile_fn=transpile_fn,
        transpile_options={'initial_layout': physical_qubits, 'optimization_level': 1},
        shots=shots)
    
    if ideal_cost is not None:
        minimizer.callbacks_sweep.append(ideal_cost.callback_sweep)
        
    if publish_sweep_result is not None:
        minimizer.callbacks_sweep.append(publish_sweep_result)

    if compiler_circuit.num_qubits == 4:
        switcher = SectionGenSwitcher(FitSymmetric, [4, 6, 7, 8, 9], 0.015)
        minimizer.callbacks_sweep.append(switcher.callback_sweep)

    if resume_from is None or 'current_sweep' not in resume_from:
        initial_params = np.ones(len(compiler_circuit.parameters)) * np.pi / 4.
        param_val = minimizer.minimize(initial_params, max_sweeps=max_sweeps)
    else:
        param_val = minimizer.resume_minimization(
            resume_from['param_val'],
            resume_from['current_sweep'],
            resume_from['current_cost'],
            resume_from['total_shots'],
            max_sweeps=max_sweeps)

    return param_val

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
    api_token = kwargs.get('api_token', None) # required as a Runtime program but not for local tests
    physical_qubits = kwargs.get('physical_qubits', None)
    error_matrix = kwargs.get('error_matrix', None)
    max_sweeps = kwargs.get('max_sweeps', 100)
    minimizer_shots = kwargs.get('minimizer_shots', 4096)
    forward_shots = kwargs.get('foward_shots', 2 * 8192)
    resume_from = kwargs.get('resume_from', None)
    
    if error_matrix is not None:
        _, state_labels = complete_meas_cal(qubit_list=list(range(num_sites)), qr=QuantumRegister(num_sites), circlabel='mcal')
        error_mitigation_filter = MeasurementFilter(error_matrix, state_labels)
    else:
        error_mitigation_filter = None

    if api_token is None:
        ibmq_backend = backend
    else:
        try:
            ibmq_provider = IBMQ.enable_account(api_token, hub='ibm-q-utokyo', group='internal', project='icepp')
        except IBMQAccountError:
            ibmq_provider = IBMQ.get_provider(hub='ibm-q-utokyo', group='internal', project='icepp')

        ibmq_backend = ibmq_provider.get_backend(backend.name())
        
    forward_step_circuits = make_step_circuits(num_sites, aJ, am, omegadt, ibmq_backend, physical_qubits)
    if not ibmq_backend.configuration().simulator:
        sim_forward_step_circuits = make_step_circuits(num_sites, aJ, am, omegadt, Aer.get_backend('statevector_simulator'))
    
    if num_sites == 2:
        approximator = make_pnp_ansatz(
            num_qubits=num_sites,
            num_layers=num_sites // 2,
            initial_x_positions=[0])
    elif num_sites == 4:
        approximator = make_pnp_ansatz(
            num_qubits=num_sites,
            num_layers=num_sites // 2,
            initial_x_positions=[1, 2],
            structure=[(1, 2), (0, 1), (2, 3)],
            first_layer_structure=[(0, 1), (2, 3)])
    
    full_forward_counts = []
    optimal_params_list = []

    if resume_from is None:
        first_rqd_step = 0
    else:
        first_rqd_step = resume_from['rqd_step']
        if first_rqd_step != 0:
            optimal_params = resume_from['optimal_params']

    for it in range(first_rqd_step, num_tsteps // tsteps_per_rqd):
        logger.info('Starting RQD step {}'.format(it))
        
        if it == 0:
            initial_state = None
        else:
            params_dict = dict(zip(approximator.parameters, optimal_params))
            initial_state = approximator.bind_parameters(params_dict)

        target_circuits = trotter_step_circuits(tsteps_per_rqd, forward_step_circuits, initial_state=initial_state, measure=False)
        
        if resume_from is None or 'current_sweep' not in resume_from:
            forward_counts = run_forward_circuits(target_circuits, backend, initial_layout=physical_qubits, shots=forward_shots, error_mitigation_filter=error_mitigation_filter)
            user_messenger.publish({'rqd_step': it, 'forward_counts': forward_counts})
            full_forward_counts += forward_counts
        
        compiler_circuit = target_circuits[-1].compose(approximator.inverse(), inplace=False)

        if ibmq_backend.configuration().simulator:
            ideal_cost = IdealCost(compiler_circuit)
        else:
            targets = trotter_step_circuits(tsteps_per_rqd, sim_forward_step_circuits, initial_state=initial_state, measure=False)
            ideal_cost = IdealCost(targets[-1].compose(approximator.inverse(), inplace=False))
            
        def callback_publish_sweep_result(minimizer, arg):
            sweep_result = {
                'rqd_step': it,
                'isweep': arg['isweep'],
                'param_val': arg['sweep_param_val'],
                'cost': arg['sweep_cost'], 
                'total_shots': arg['current_shots'] + arg['sweep_shots'],
                'shots_values': np.array(ideal_cost.shots_sweep),
                'cost_values': np.array(ideal_cost.costs_sweep)
            }
            user_messenger.publish(sweep_result)
        
        optimal_params = run_fisc(
            compiler_circuit,
            backend,
            physical_qubits,
            max_sweeps,
            minimizer_shots, 
            error_mitigation_filter,
            ideal_cost,
            callback_publish_sweep_result,
            resume_from
        )
        user_messenger.publish({'rqd_step': it, 'optimal_params': optimal_params})
        optimal_params_list.append(optimal_params)
        
        logger.info('Completed RQD step {}'.format(it))

    return full_forward_counts, optimal_params_list
