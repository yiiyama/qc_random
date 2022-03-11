import numpy as np
from phi_dependence_experiment import PhaseDependenceExperiment

from qiskit_private_tools import backends_table, my_backends, my_providers

backends_table()

experiment_enabled = set()
for provider in my_providers.values():
    experiment_enabled.update(spec['name'] for spec in provider.service('experiment').backends())

for name, backend in my_backends.items():
    if name in ['ibmq_quito'] or name not in experiment_enabled:
        continue
        
    print('Measuring phase dependency of {}'.format(name))
        
    exp = PhaseDependenceExperiment(list(range(backend.configuration().n_qubits)), np.linspace(-np.pi, np.pi, 20))
    exp_data = exp.run(backend=backend, shots=1024)
    exp_data.block_for_results()
    exp_data.save()
