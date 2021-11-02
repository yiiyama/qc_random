import collections
from qiskit import IBMQ

MY_PROVIDERS = collections.OrderedDict([('icepp', ('ibm-q-utokyo', 'internal', 'icepp')), ('ibmqr', ('ibm-q-research', 'tokyo-1', 'main'))])
my_providers = dict()
my_backends = dict()

def backends_table(pulse_only=False):
    if IBMQ.active_account() is None:
        IBMQ.load_account()

    provider_lists = dict()
    
    for provname, provspec in MY_PROVIDERS.items():
        provider = IBMQ.get_provider(*provspec)
        my_providers['/'.join(provspec)] = provider
        for backend in provider.backends():
            config = backend.configuration()

            try:
                if config.simulator:
                    continue
                    
                if not backend.status().operational:
                    continue
            except:
                continue

            if pulse_only and not config.open_pulse:
                continue

            try:
                other = my_backends[backend.name()]
            except KeyError:
                my_backends[backend.name()] = backend
            else:
                other_config = other.configuration()
                if config.max_experiments * config.max_shots > other_config.max_experiments * other_config.max_shots:
                    my_backends[backend.name()] = backend
                
            try:
                provider_lists[backend.name()].append(provname)
            except KeyError:
                provider_lists[backend.name()] = [provname]

    print('My backends:')
    print('+{}-{}-{}-{}-{}-{}-{}-{}-{}+'.format('-' * 15, '-' * 3, '-' * 4, '-' * 16, '-' * 5, '-' * 5, '-' * 4, '-' * 4, '-' * 15))
    print('|{name:15s}|{nq:3s}|{qv:4s}|{chip:16s}|{pulse:5s}|{maxshots:5s}|{maxexps:4s}|{jobs:4s}|{providers:15s}|'.format(
        name='Name',
        nq='Nq',
        qv='QV',
        chip='Chip',
        pulse='Pulse',
        maxshots='Shots',
        maxexps='Exps',
        jobs='Jobs',
        providers='Providers'
    ))
    print('+{}+{}+{}+{}+{}+{}+{}+{}+{}+'.format('-' * 15, '-' * 3, '-' * 4, '-' * 16, '-' * 5, '-' * 5, '-' * 4, '-' * 4, '-' * 15))
    for backend in sorted(my_backends.values(), key=(lambda backend: -backend.configuration().n_qubits)):
        config = backend.configuration()
        status = backend.status()
        
        try:
            chip = '{} {:.2f}'.format(config.processor_type['family'], config.processor_type['revision'])
        except ValueError:
            chip = '{} {}'.format(config.processor_type['family'], config.processor_type['revision'])
        
        print('|{name:15s}|{nq:3d}|{qv:4d}|{chip:16s}|{pulse:5s}|{maxshots:5d}|{maxexps:4d}|{jobs:4d}|{providers:15s}|'.format(
            name=backend.name(),
            nq=config.n_qubits,
            qv=0,
            chip=chip,
            pulse=('Yes' if config.open_pulse else 'No'),
            maxshots=config.max_shots,
            maxexps=config.max_experiments,
            jobs=status.pending_jobs,
            providers=' '.join(provider_lists[backend.name()])
        ))
        print('+{}+{}+{}+{}+{}+{}+{}+{}+{}+'.format('-' * 15, '-' * 3, '-' * 4, '-' * 16, '-' * 5, '-' * 5, '-' * 4, '-' * 4, '-' * 15))
