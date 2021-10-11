import re
import collections
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as scispec
import scipy.optimize as sciopt
from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit import Gate
from qiskit.result import Result
from qiskit.test.mock import FakeValencia
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter, MeasurementFilter
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData, FitVal
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.curve_analysis import plot_curve_fit, plot_errorbar, curve_fit
#from qiskit_experiments.curve_analysis.curve_fit import process_curve_data
from qiskit_experiments.curve_analysis.data_processing  import level2_probability, filter_data

def get_instruction_by_name(schedule, pattern):
    return next(inst for _, inst in schedule.instructions if inst.name is not None and re.match(pattern, inst.name))

def get_closest_multiple_of_16(num):
    return int(np.round(num / 16.)) * 16

def find_native_cr_direction(qubits, backend):
    cx_schedule = backend.defaults().instruction_schedule_map.get('cx', qubits)
    gs_on_drive_channel = get_instruction_by_name(cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$')
    drive_channel = re.match(r'CR90p_(d[0-9]+)_u[0-9]+$', gs_on_drive_channel.name).group(1)
    x_qubit = backend.configuration().channels[drive_channel]['operates']['qubits'][0]
    z_qubit = qubits[0] if qubits[1] == x_qubit else qubits[1]

    return (z_qubit, x_qubit)

def resolve_cx(backend, control_qubit, target_qubit):
    schedule = backend.defaults().instruction_schedule_map.get('cx', (control_qubit, target_qubit))
    
    cr_start_time = next(t for t, inst in schedule.instructions if type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel)
    last_cr_start, last_cr = next((t, inst) for t, inst in reversed(schedule.instructions) if type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel)
    cr_end_time = last_cr_start + last_cr.pulse.duration

    pre_sched = schedule.filter(time_ranges=[(0, cr_start_time)])
    core_sched = pulse.Schedule(schedule.filter(instruction_types=[pulse.Play], time_ranges=[(cr_start_time, cr_end_time)]).shift(-cr_start_time), name='cx_core_{}_{}'.format(control_qubit, target_qubit))
    post_sched = schedule.filter(time_ranges=[(cr_end_time, schedule.duration)])
    
    channel_logic_map = {}
    for ch_name, ch_config in backend.configuration().channels.items():
        if ch_config['operates']['qubits'] == [control_qubit]:
            channel_logic_map[ch_name] = 0
        elif ch_config['operates']['qubits'] == [target_qubit]:
            channel_logic_map[ch_name] = 1
            
    def schedule_to_circuit(sched):
        circuit = QuantumCircuit(2)

        for _, inst in sched.instructions:
            try:
                qubit = channel_logic_map[inst.channel.name]
            except KeyError:
                continue
            
            if type(inst) is pulse.ShiftPhase and type(inst.channel) is pulse.DriveChannel:
                circuit.rz(-inst.phase, qubit)
            elif type(inst) is pulse.Play:
                matches = re.match('(X|Y)(|90)(p|m)_', inst.name)
                if not matches:
                    continue
                if matches.group(3) == 'm':
                    circuit.rz(np.pi, qubit)
                if matches.group(1) == 'Y':
                    circuit.rz(-np.pi / 2., qubit)
                if matches.group(2) == '':
                    circuit.x(qubit)
                else:
                    circuit.sx(qubit)
                if matches.group(1) == 'Y':
                    circuit.rz(np.pi / 2., qubit)
                if matches.group(3) == 'm':
                    circuit.rz(-np.pi, qubit)
                    
        return circuit

    pre_circ = schedule_to_circuit(pre_sched)
    post_circ = schedule_to_circuit(post_sched)
    return pre_circ, core_sched, post_circ

def cx_circuit(backend, control_qubit, target_qubit):
    pre_circ, core_sched, post_circ = resolve_cx(backend, control_qubit, target_qubit)
    
    core_gate = Gate(core_sched.name, 2, [])
    
    circuit = QuantumCircuit(2)
    circuit.compose(pre_circ, inplace=True)
    circuit.append(core_gate, (0, 1))
    circuit.compose(post_circ, inplace=True)

    circuit.add_calibration(core_gate.name, (control_qubit, target_qubit), core_sched)

    return circuit

class MeasurementErrorAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options()
    
    def _run_analysis(self, experiment_data, parameter_guess=None, plot=True, ax=None):
        state_labels = []
        for datum in experiment_data.data():
            state_label = datum['metadata']['state_label']
            if state_label in state_labels:
                break
            state_labels.append(state_label)

        meas_fitter = CompleteMeasFitter(None, state_labels, circlabel='mcal')
        
        nstates = len(state_labels)

        for job_id in experiment_data.job_ids:
            full_result = experiment_data.backend.retrieve_job(job_id).result()
            # full_result might contain repeated experiments
            for iset in range(len(full_result.results) // nstates):
                try:
                    date = full_result.date
                except:
                    date = None
                try:
                    status = full_result.status
                except:
                    status = None
                try:
                    header = full_result.header
                except:
                    header = None
                    
                result = Result(full_result.backend_name, full_result.backend_version, \
                                full_result.qobj_id, full_result.job_id, \
                                full_result.success, full_result.results[iset * nstates:(iset + 1) * nstates], \
                                date=date, status=status, header=header, **full_result._metadata)

                meas_fitter.add_data(result)
        
        results = [
            AnalysisResultData('error_matrix', meas_fitter.cal_matrix, extra=state_labels)
        ]
                
        plots = []
        if plot:
            figure, ax = plt.subplots(1, 1)
            meas_fitter.plot_calibration(ax=ax)
            plots.append(figure)
        
        return results, plots
    
class MeasurementErrorExperiment(BaseExperiment):
    __analysis_class__ = MeasurementErrorAnalysis
    
    def __init__(self, qubit_list, circuits_per_state=1):
        super().__init__(qubit_list)
        
        self.circuits_per_state = circuits_per_state

    def circuits(self, backend=None):
        if backend is None:
            backend = FakeValencia()
            print('Using FakeValencia for backend')
            
        qreg = QuantumRegister(len(self.physical_qubits))

        circuits, state_labels = complete_meas_cal(qubit_list=list(range(qreg.size)), qr=qreg, circlabel='mcal')
        for circuit, state_label in zip(circuits, state_labels):
            circuit.metadata = {
                'experiment_type': self._type,
                'physical_qubits': self.physical_qubits,
                'state_label': state_label
            }

        return circuits * self.circuits_per_state
    
class MeasurementErrorMitigation(object):
    def __init__(self, backend, qubits):
        self.backend = backend
        self.qubits = qubits
        self.filter = None
        
    def run_experiment(self, circuits_per_state=1):
        exp = MeasurementErrorExperiment(self.qubits, circuits_per_state=circuits_per_state)
        exp_data = exp.run(backend=self.backend, shots=self.backend.configuration().max_shots)
        print('Experiment ID:', exp_data.experiment_id)
        exp_data.block_for_results()
        exp_data.save()
        self._load_from_exp_data(exp_data)
        
    def load_matrix(self, experiment_id):
        exp_data = DbExperimentData.load(experiment_id, self.backend.provider().service("experiment"))
        self._load_from_exp_data(exp_data)
        
    def _load_from_exp_data(self, exp_data):
        analysis_result = exp_data.analysis_results()[0]
        self.filter = MeasurementFilter(analysis_result.value, analysis_result.extra)

    def apply(self, counts_list):
        corrected_counts = []
        for counts in counts_list:
            corrected_counts.append(self.filter.apply(counts))
        
        return corrected_counts


class LinearizedCRRabiAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options(
            parameter_guess={'alpha': np.pi * 0.5 / 500., 'phi0': 0., 'amp': 0.5, 'offset': 0.5}
        )
    
    def _run_analysis(self, experiment_data, parameter_guess=None, plot=True, ax=None):
        data = experiment_data.data()
        metadata = data[0]['metadata']
        
        counts = collections.defaultdict(int)
        totals = collections.defaultdict(int)
    
        for datum in data:
            width = datum['metadata']['xval']
            counts[width] += datum['counts'].get('00', 0)
            totals[width] += sum(datum['counts'].values())

        xdata = np.array(sorted(counts.keys()), dtype=float)
        ydata = np.array([counts[w] for w in sorted(counts.keys())], dtype=float)
        total = np.array([totals[w] for w in sorted(counts.keys())], dtype=float)
        ydata /= total
        ysigma = np.sqrt(ydata * (1. - ydata) / total)
        
        if parameter_guess is None:
            p0 = (np.pi * 0.5 / 500., 0., 0.5, 0.5)
        else:
            p0 = (parameter_guess['alpha'], parameter_guess['phi0'], parameter_guess['amp'], parameter_guess['offset'])
            
        def fun(x, alpha, phi0, amp, offset):
            return offset + amp * np.cos(alpha * x + phi0)
            
        fit_result = curve_fit(fun, xdata, ydata, p0, sigma=ysigma)
        
        if np.abs(fit_result.popt[1]) < np.pi / 4. and np.abs(fit_result.popt[2] - 0.5) < 0.1 and np.abs(fit_result.popt[3] - 0.5) < 0.1:
            quality = 'good'
        else:
            quality = 'bad'
            
        summary = {
            'shots_per_point': np.sum(total) / xdata.shape[0],
            'fit_result': fit_result
        }
        
        results = [
            AnalysisResultData('alpha', FitVal(fit_result.popt[0], fit_result.popt_err[0])),
            AnalysisResultData('phi0', FitVal(fit_result.popt[1], fit_result.popt_err[1])),
            AnalysisResultData('amp', FitVal(fit_result.popt[2], fit_result.popt_err[2])),
            AnalysisResultData('offset', FitVal(fit_result.popt[3], fit_result.popt_err[3])),
            AnalysisResultData('summary', summary, chisq=fit_result.reduced_chisq, quality=quality)
        ]

        plots = []
        if plot:
            ax = plot_curve_fit(fun, fit_result, ax=ax, fit_uncertainty=True)
            ax = plot_errorbar(xdata, ydata, ysigma, ax=ax)
            ax.tick_params(labelsize=14)
            ax.set_title('Rzx[{},{}]'.format(metadata['z_qubit'], metadata['x_qubit']))
            ax.set_xlabel('GaussianSquare width', fontsize=16)
            ax.set_ylabel('P(00)', fontsize=16)
            ax.grid(True)
            plots.append(ax.get_figure())
        
        return results, plots

class LinearizedCRRabiExperiment(BaseExperiment):
    __analysis_class__ = LinearizedCRRabiAnalysis
    
    def __init__(self, qubits, backend, max_width=1000, step_size=32, randomize_width=True, circuits_per_point=1):
        super().__init__(find_native_cr_direction(qubits, backend))
        
        self.width_values = np.arange(0, max_width + 1, step_size)
        if randomize_width:
            for i in range(self.width_values.shape[0]):
                self.width_values[i] += np.random.randint(-(step_size // 2), step_size // 2)
                
        self.circuits_per_point = circuits_per_point
                
    def _additional_metadata(self):
        return [('widths', self.width_values.tolist())]

    def circuits(self, backend=None):
        if backend is None:
            backend = FakeValencia()
            print('Using FakeValencia for backend')
            
        cr = LinearizedCR(backend, self.physical_qubits)
        
        circuits = []
        
        for width in self.width_values:
            circuit = cr.rzx_circuit(width)
            circuit.measure_all()
            circuit.metadata = {
                'experiment_type': self._type,
                'z_qubit': cr.z_qubit,
                'x_qubit': cr.x_qubit,
                'xval': width
            }            
            
            circuits.append(circuit)
        
        return circuits * self.circuits_per_point


class BaseCR(object):
    def __init__(self, backend, qubits):
        self.backend = backend
        
        qubits = find_native_cr_direction(qubits, backend)
        
        self.z_qubit = qubits[0]
        self.x_qubit = qubits[1]
        
        self.cx_schedule = backend.defaults().instruction_schedule_map.get('cx', qubits)

    def rzx_circuit(self, phi_value, with_schedule=False):
        cr_core_sched = self.get_cr_core_schedule(phi_value)
        
        circuit = QuantumCircuit(QuantumRegister(2))

        if phi_value == 0.:
            if with_schedule:
                return circuit, cr_core_sched
            else:
                return circuit

        gate_name = 'cr_core_{}_{}'.format(self.z_qubit, self.x_qubit)
        cr_core_gate = Gate(gate_name, 2, [])

        circuit.x(0)
        circuit.append(cr_core_gate, (0, 1))
        
        circuit.add_calibration(gate_name, (self.z_qubit, self.x_qubit), cr_core_sched)
        circuit.add_calibration(gate_name, (self.x_qubit, self.z_qubit), cr_core_sched)

        if with_schedule:
            return circuit, cr_core_sched
        else:
            return circuit

    def rzz_circuit(self, phi_value, with_schedule=False):
        rzx = self.rzx_circuit(phi_value, with_schedule=with_schedule)
    
        if with_schedule:
            rzx_circuit, cr_core_sched = rzx
        else:
            rzx_circuit = rzx
        
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(1)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.h(1)
    
        if with_schedule:
            return circuit, cr_core_sched
        else:
            return circuit
    
    def rxx_circuit(self, phi_value, with_schedule=False):
        rzx = self.rzx_circuit(phi_value, with_schedule=with_schedule)
    
        if with_schedule:
            rzx_circuit, cr_core_sched = rzx
        else:
            rzx_circuit = rzx
        
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(0)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.h(0)
    
        if with_schedule:
            return circuit, cr_core_sched
        else:
            return circuit

    def ryy_circuit(self, phi_value, with_schedule=False):
        rzx = self.rzx_circuit(phi_value, with_schedule=with_schedule)
    
        if with_schedule:
            rzx_circuit, cr_core_sched = rzx
        else:
            rzx_circuit = rzx
        
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.sdg(0)
        circuit.h(0)
        circuit.sdg(1)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.s(1)
        circuit.h(0)
        circuit.s(0)
    
        if with_schedule:
            return circuit, cr_core_sched
        else:
            return circuit
        
    def get_z_qubit_drive_channel(self):
        instructions = self.cx_schedule.instructions
        return next(inst.channel for _, inst in instructions if (type(inst) is pulse.ShiftPhase and type(inst.channel) is pulse.DriveChannel))
    
    def get_x_qubit_drive_channel(self):
        instructions = self.cx_schedule.instructions
        return get_instruction_by_name(self.cx_schedule, r'X90p_d[0-9]+$').channel # Drag
    
    def get_control_channel(self):
        instructions = self.cx_schedule.instructions
        return next(inst.channel for _, inst in instructions if (type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel))


class LinearizedCR(BaseCR):
    def __init__(self, backend, qubits):
        super().__init__(backend, qubits)
        
        self.alpha = 1.
        self.phi0 = 0.
        
    def load_calibration(self, experiment_id):
        load_exp = DbExperimentData.load(experiment_id, self.backend.provider().service("experiment"))
        self.alpha = next(res for res in load_exp.analysis_results() if res.name == 'alpha').value.value
        self.phi0 = next(res for res in load_exp.analysis_results() if res.name == 'phi0').value.value
    
    def get_cr_core_schedule(self, phi_value):
        width = (phi_value - self.phi0) / self.alpha
        
        z_qubit_drive = self.get_z_qubit_drive_channel()
        x_qubit_drive = self.get_x_qubit_drive_channel()
        control_drive = self.get_control_channel()

        x_pulse = get_instruction_by_name(self.cx_schedule, r'Xp_d[0-9]+$').pulse
        cx_cr_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_u[0-9]+$').pulse
        cx_rotary_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse

        if width == 0.:
            with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_core_{}_{}'.format(self.z_qubit, self.x_qubit)) as cr_core_sched:
                pulse.play(x_pulse, z_qubit_drive)

            return cr_core_sched

        cr_amp = cx_cr_pulse.amp
        crr_amp = cx_rotary_pulse.amp
        cr_sigma = cx_cr_pulse.sigma
        cr_flank_width = (cx_cr_pulse.duration - cx_cr_pulse.width) // 2

        # smallest multiple of 16 greater than |width| + 2 * (original flank width)
        #gs_duration = int(np.ceil((np.abs(width) + 2. * cr_flank_width) / 16.) * 16)
        gs_duration = get_closest_multiple_of_16(np.abs(width) + 2 * cr_flank_width)
        target_flank_width = (gs_duration - np.abs(width)) / 2.

        def area(sigma, flank_width):
            n_over_s2 = flank_width / sigma / np.sqrt(2.)
            gaus_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(n_over_s2)
            pedestal = np.exp(-n_over_s2 * n_over_s2)
            pedestal_integral = pedestal * flank_width
            return (gaus_integral - pedestal_integral) / (1. - pedestal)

        def diff_area(sigma, flank_width):
            n = flank_width / sigma
            n_over_s2 = n / np.sqrt(2.)
            gaus_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(n_over_s2)
            pedestal = np.exp(-n_over_s2 * n_over_s2)
            pedestal_integral = pedestal * flank_width
            diff_gaus_integral = gaus_integral / sigma - n * pedestal
            diff_pedestal_integral = n * n * n * pedestal
            return (diff_gaus_integral - diff_pedestal_integral - (diff_gaus_integral * pedestal_integral - gaus_integral * diff_pedestal_integral) / flank_width) / (1. - pedestal) / (1. - pedestal)

        cr_flank_area = area(cr_sigma, cr_flank_width)

        def func(sigma):
            return area(sigma, target_flank_width) - cr_flank_area

        def fprime(sigma):
            return diff_area(sigma, target_flank_width)

        gs_sigma = sciopt.newton(func, cr_sigma, fprime)
        
        phi_label = int(np.round(phi_value / np.pi * 180.))

        def gaus_sq(amp, rotary):
            if rotary:
                name = 'CRGS{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index)
            else:
                name = 'CRGS{}p_u{}'.format(phi_label, control_drive.index)

            return pulse.GaussianSquare(duration=gs_duration, amp=amp, sigma=gs_sigma, width=np.abs(width), name=name)

        def gaus(amp, rotary):
            if rotary:
                name = 'CRG{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index)
            else:
                name = 'CRG{}p_u{}'.format(phi_label, control_drive.index)

            return pulse.Gaussian(duration=(2 * cr_flank_width), amp=amp, sigma=cr_sigma, name=name)

        if width > 0.:
            forward = gaus_sq
            cancel = gaus
        else:
            forward = gaus
            cancel = gaus_sq

        cr_pulse = forward(cr_amp, False)
        cr_rotary_pulse = forward(crr_amp, True)
        cr_echo = forward(-cr_amp, False)
        cr_rotary_echo = forward(-crr_amp, True)

        cancel_pulse = cancel(-cr_amp, False)
        cancel_rotary_pulse = cancel(-crr_amp, True)
        cancel_echo = cancel(cr_amp, False)
        cancel_rotary_echo = cancel(crr_amp, True)

        with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_core_{}_{}'.format(self.z_qubit, self.x_qubit)) as cr_core_sched:
            ## echo (without the first X on control)
            with pulse.align_left():
                pulse.play(cr_echo, control_drive)
                pulse.play(cancel_echo, control_drive)
                pulse.play(cr_rotary_echo, x_qubit_drive)
                pulse.play(cancel_rotary_echo, x_qubit_drive)

            pulse.play(x_pulse, z_qubit_drive)

            ## forward
            with pulse.align_left():
                pulse.play(cr_pulse, control_drive)
                pulse.play(cancel_pulse, control_drive)
                pulse.play(cr_rotary_pulse, x_qubit_drive)
                pulse.play(cancel_rotary_pulse, x_qubit_drive)

        return cr_core_sched
    
    
class PulseEfficientCR(BaseCR):
    def get_cr_core_schedule(self, phi_value):
        z_qubit_drive = self.get_z_qubit_drive_channel()
        x_qubit_drive = self.get_x_qubit_drive_channel()
        control_drive = self.get_control_channel()

        x_pulse = get_instruction_by_name(self.cx_schedule, r'Xp_d[0-9]+$').pulse
        cx_cr_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_u[0-9]+$').pulse
        cx_rotary_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse    

        if phi_value == 0.:
            with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
                pulse.play(x_pulse, z_qubit_drive)

            return cr_core_sched

        cr_amp = cx_cr_pulse.amp
        crr_amp = cx_rotary_pulse.amp
        sigma = cx_cr_pulse.sigma
        flank_width = (cx_cr_pulse.duration - cx_cr_pulse.width) // 2

        normal_flank_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(flank_width / np.sqrt(2.) / sigma)
        pedestal = np.exp(-0.5 * np.square(flank_width / sigma))
        grounded_flank_integral = (normal_flank_integral - pedestal * flank_width) / (1. - pedestal)
        flank_area = np.abs(cr_amp) * grounded_flank_integral
        cr45_area_norm = np.abs(cr_amp) * cx_cr_pulse.width + 2. * flank_area
        minimum_phi = 2. * np.pi / 4. * (2. * flank_area) / cr45_area_norm

        phi_label = int(np.round(phi_value / np.pi * 180.))
        
        if phi_value <= minimum_phi:
            amp_ratio = phi_value / minimum_phi
            duration = 2 * flank_width
            cr_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * cr_amp), sigma=sigma, name='CR{}p_u{}'.format(phi_label, control_drive.index))
            cr_rotary_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * crr_amp), sigma=sigma, name='CR{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
            cr_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * cr_amp), sigma=sigma, name='CR{}m_u{}'.format(phi_label, control_drive.index))
            cr_rotary_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * crr_amp), sigma=sigma, name='CR{}m_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
        else:
            area = phi_value / 2. / (np.pi / 4.) * cr45_area_norm
            width = (area - 2. * flank_area) / np.abs(cr_amp)
            duration = get_closest_multiple_of_16(width + 2 * flank_width)
            cr_pulse = pulse.GaussianSquare(duration=duration, amp=cr_amp, sigma=sigma, width=width, name='CR{}p_u{}'.format(phi_label, control_drive.index))
            cr_rotary_pulse = pulse.GaussianSquare(duration=duration, amp=crr_amp, sigma=sigma, width=width, name='CR{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
            cr_echo = pulse.GaussianSquare(duration=duration, amp=-cr_amp, sigma=sigma, width=width, name='CR{}m_u{}'.format(phi_label, control_drive.index))
            cr_rotary_echo = pulse.GaussianSquare(duration=duration, amp=-crr_amp, sigma=sigma, width=width, name='CR{}m_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))

        with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
            ## echo (without the first X on control)
            with pulse.align_left():
                pulse.play(cr_echo, control_drive, name=cr_echo.name)
                pulse.play(cr_rotary_echo, x_qubit_drive, name=cr_rotary_echo.name)

            pulse.play(x_pulse, z_qubit_drive, name=x_pulse.name)

            ## forward
            with pulse.align_left():
                pulse.play(cr_pulse, control_drive, name=cr_pulse.name)
                pulse.play(cr_rotary_pulse, x_qubit_drive, name=cr_rotary_pulse.name)

        return cr_core_sched
