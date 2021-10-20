import numpy as np
from qiskit import IBMQ
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.framework import ExperimentData

from cross_resonance import LinearizedCR, PulseEfficientCR, DefaultRtt
from process_fidelity import RzzFidelityExperiment

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research', group='tokyo-1', project='main')
backend = provider.get_backend('ibmq_quito')

physical_qubits = [2, 1]

mem_exp_id = 'da7ccf40-d154-45c6-91f4-52153de9d615'

rtt_li = LinearizedCR(backend, physical_qubits)
rtt_li.load_calibration('8b75a6fa-acfd-4c5f-a007-e01e0e8935b0')

rtt_pe = PulseEfficientCR(backend, physical_qubits)

rtt_def = DefaultRtt()

phi_values = np.concatenate((np.linspace(np.pi / 60., np.pi / 6., 4, endpoint=False), np.linspace(np.pi / 6., np.pi, 5)))

exp = RzzFidelityExperiment(physical_qubits, rtt_li, phi_values, error_mitigation_exp_id=mem_exp_id)
exp_data = exp.run(backend=backend, shots=8192)
print(exp_data.experiment_id)
exp_data.block_for_results()
exp_data.save()

exp = RzzFidelityExperiment(physical_qubits, rtt_pe, phi_values, error_mitigation_exp_id=mem_exp_id)
exp_data = exp.run(backend=backend, shots=8192)
print(exp_data.experiment_id)
exp_data.block_for_results()
exp_data.save()

exp = RzzFidelityExperiment(physical_qubits, rtt_def, phi_values, error_mitigation_exp_id=mem_exp_id)
exp_data = exp.run(backend=backend, shots=8192)
print(exp_data.experiment_id)
exp_data.block_for_results()
exp_data.save()