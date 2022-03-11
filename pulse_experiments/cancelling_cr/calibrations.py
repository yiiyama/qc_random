import numpy as np
import scipy
import scipy.special as scispec
import scipy.optimize as sciopt
from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit import Gate

from pulse_experiment_utils import get_instruction_by_name, get_closest_multiple_of_16

class PaddedGaussianSquare(pulse.GaussianSquare):
    def __init__(self, pulse_duration, amp, sigma, width, name=None):
        residual = pulse_duration % 16
        if residual == 0:
            duration = pulse_duration
        else:
            duration = pulse_duration + 16 - residual
            
        pulse.GaussianSquare.__init__(self, duration, amp, sigma, width=width, name=name)
        self._risefall_sigma_ratio = (pulse_duration - self.width) / (2.0 * self.sigma)
        
        self.pulse_duration = pulse_duration

    def get_waveform(self):
        waveform = pulse.GaussianSquare.get_waveform(self)
        residual = self.pulse_duration % 16
        if residual != 0:
            waveform._samples = np.concatenate((np.zeros(16 - residual, dtype=waveform._samples.dtype), waveform._samples))

        return waveform
    
    
def pulse_efficient_cr_core_schedule(phi_value, cx_schedule, backend):
    instructions = cx_schedule.instructions

    control_qubit_drive = instructions[0][1].channel # ShiftPhase
    control_qubit = control_qubit_drive.index
    target_qubit_drive = instructions[5][1].channel # Drag
    target_qubit = target_qubit_drive.index
    control_drive = next(inst[1].channel for inst in instructions if (type(inst[1]) is pulse.Play and type(inst[1].channel) is pulse.ControlChannel))
    
    x_pulse = get_instruction_by_name(cx_schedule, r'Xp_d[0-9]+$').pulse
    cx_cr_pulse = get_instruction_by_name(cx_schedule, r'CR90p_u[0-9]+$').pulse
    cx_rotary_pulse = get_instruction_by_name(cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse    

    if phi_value == 0.:
        with pulse.build(backend=backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
            pulse.play(x_pulse, control_qubit_drive)
            
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

    phi_label = np.round(phi_value / np.pi * 180.)
    if phi_value <= minimum_phi:
        amp_ratio = phi_value / minimum_phi
        duration = 2 * flank_width
        cr_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * cr_amp), sigma=sigma, name='CR{}p_u{}'.format(phi_label, control_drive.index))
        cr_rotary_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * crr_amp), sigma=sigma, name='CR{}p_d{}_u{}'.format(phi_label, target_qubit, control_drive.index))
        cr_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * cr_amp), sigma=sigma, name='CR{}m_u{}'.format(phi_label, control_drive.index))
        cr_rotary_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * crr_amp), sigma=sigma, name='CR{}m_d{}_u{}'.format(phi_label, target_qubit, control_drive.index))
    else:
        area = phi_value / 2. / (np.pi / 4.) * cr45_area_norm
        width = (area - 2. * flank_area) / np.abs(cr_amp)
        duration = get_closest_multiple_of_16(width + 2 * flank_width)
        cr_pulse = pulse.GaussianSquare(duration=duration, amp=cr_amp, sigma=sigma, width=width, name='CR{}p_u{}'.format(phi_label, control_drive.index))
        cr_rotary_pulse = pulse.GaussianSquare(duration=duration, amp=crr_amp, sigma=sigma, width=width, name='CR{}p_d{}_u{}'.format(phi_label, target_qubit, control_drive.index))
        cr_echo = pulse.GaussianSquare(duration=duration, amp=-cr_amp, sigma=sigma, width=width, name='CR{}m_u{}'.format(phi_label, control_drive.index))
        cr_rotary_echo = pulse.GaussianSquare(duration=duration, amp=-crr_amp, sigma=sigma, width=width, name='CR{}m_d{}_u{}'.format(phi_label, target_qubit, control_drive.index))
    
    with pulse.build(backend=backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
        ## echo (without the first X on control)
        with pulse.align_left():
            pulse.play(cr_echo, control_drive, name=cr_echo.name)
            pulse.play(cr_rotary_echo, target_qubit_drive, name=cr_rotary_echo.name)

        pulse.play(x_pulse, control_qubit_drive, name=x_pulse.name)

        ## forward
        with pulse.align_left():
            pulse.play(cr_pulse, control_drive, name=cr_pulse.name)
            pulse.play(cr_rotary_pulse, target_qubit_drive, name=cr_rotary_pulse.name)
            
    return cr_core_sched

def pulse_efficient_rzx(phi_value, cx_schedule, backend, with_schedule=False):
    register = QuantumRegister(2)
    
    if phi_value == 0.:
        circuit = QuantumCircuit(register)

        if with_schedule:
            return circuit, pulse.Schedule(name='cr_gate_core')
        else:
            return circuit

    instructions = cx_schedule.instructions    

    control_qubit_drive = next(inst.channel for _, inst in instructions if (type(inst) is pulse.ShiftPhase and type(inst.channel) is pulse.DriveChannel))
    control_qubit = control_qubit_drive.index
    target_qubit_drive = get_instruction_by_name(cx_schedule, r'X90p_d[0-9]+$').channel # Drag
    target_qubit = target_qubit_drive.index
    
    cx_cr_pulse = get_instruction_by_name(cx_schedule, r'CR90p_u[0-9]+$').pulse
    
    cr_core_sched = pulse_efficient_cr_core_schedule(phi_value, cx_schedule, backend)
            
    cr_core_gate = Gate('cr_core_gate', 2, [])
    
    circuit = QuantumCircuit(register)
    circuit.x(0)
    circuit.append(cr_core_gate, (0, 1))

    circuit.add_calibration('cr_core_gate', (control_qubit, target_qubit), cr_core_sched)
    
    if with_schedule:
        return circuit, cr_core_sched
    else:
        return circuit
    
    
def linearized_cr_core_schedule(width, cx_schedule, backend, phi_label=0, adjust_sigma=False):
    instructions = cx_schedule.instructions

    control_qubit_drive = next(inst.channel for _, inst in instructions if (type(inst) is pulse.ShiftPhase and type(inst.channel) is pulse.DriveChannel))
    control_qubit = control_qubit_drive.index
    target_qubit_drive = get_instruction_by_name(cx_schedule, r'X90p_d[0-9]+$').channel # Drag
    target_qubit = target_qubit_drive.index
    control_drive = next(inst.channel for _, inst in instructions if (type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel))
    
    x_pulse = get_instruction_by_name(cx_schedule, r'Xp_d[0-9]+$').pulse
    cx_cr_pulse = get_instruction_by_name(cx_schedule, r'CR90p_u[0-9]+$').pulse
    cx_rotary_pulse = get_instruction_by_name(cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse
    
    if width == 0.:
        with pulse.build(backend=backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
            pulse.play(x_pulse, control_qubit_drive)
            
        return cr_core_sched
    
    cr_amp = cx_cr_pulse.amp
    crr_amp = cx_rotary_pulse.amp
    cr_sigma = cx_cr_pulse.sigma
    cr_flank_width = (cx_cr_pulse.duration - cx_cr_pulse.width) // 2
    
    if adjust_sigma:
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
        
        print('width', width, 'gs_duration', gs_duration, 'target', target_flank_width, 'gs_sigma', gs_sigma)
        
    else:
        gs_sigma = cr_sigma
        gs_duration = int(np.ceil((np.abs(width) + 2 * cr_flank_width) / 16.) * 16)
        
    def gaus_sq(amp, rotary):
        if rotary:
            name = 'CRGS{}p_d{}_u{}'.format(phi_label, target_qubit, control_drive.index)
        else:
            name = 'CRGS{}p_u{}'.format(phi_label, control_drive.index)
            
        return pulse.GaussianSquare(duration=gs_duration, amp=amp, sigma=gs_sigma, width=np.abs(width), name=name)
    
    def gaus(amp, rotary):
        if rotary:
            name = 'CRG{}p_d{}_u{}'.format(phi_label, target_qubit, control_drive.index)
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

    with pulse.build(backend=backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
        ## echo (without the first X on control)
        with pulse.align_left():
            pulse.play(cr_echo, control_drive)
            pulse.play(cancel_echo, control_drive)
            pulse.play(cr_rotary_echo, target_qubit_drive)
            pulse.play(cancel_rotary_echo, target_qubit_drive)

        pulse.play(x_pulse, control_qubit_drive)

        ## forward
        with pulse.align_left():
            pulse.play(cr_pulse, control_drive)
            pulse.play(cancel_pulse, control_drive)
            pulse.play(cr_rotary_pulse, target_qubit_drive)
            pulse.play(cancel_rotary_pulse, target_qubit_drive)
            
    return cr_core_sched


def linearized_rzx(phi_value, cx_schedule, phi_to_width, backend, adjust_sigma=False, with_schedule=False):
    register = QuantumRegister(2)
    
    if phi_value == 0.:
        circuit = QuantumCircuit(register)

        if with_schedule:
            return circuit, pulse.Schedule(name='cr_gate_core')
        else:
            return circuit

    instructions = cx_schedule.instructions

    control_qubit_drive = instructions[0][1].channel # ShiftPhase
    control_qubit = control_qubit_drive.index
    target_qubit_drive = instructions[5][1].channel # Drag
    target_qubit = target_qubit_drive.index
    
    cx_cr_pulse = get_instruction_by_name(cx_schedule, r'CR90p_u[0-9]+$').pulse

    width = phi_to_width(phi_value)
        
    cr_core_sched = linearized_cr_core_schedule(width, cx_schedule, backend, phi_label=int(np.round(phi_value / np.pi * 180.)), adjust_sigma=adjust_sigma)
            
    cr_core_gate = Gate('cr_core_gate', 2, [])
    
    circuit = QuantumCircuit(register)
    circuit.x(0)
    circuit.append(cr_core_gate, (0, 1))

    circuit.add_calibration('cr_core_gate', (control_qubit, target_qubit), cr_core_sched)
    
    if with_schedule:
        return circuit, cr_core_sched
    else:
        return circuit

    
def linearized_rzz(phi_value, cx_schedule, phi_to_width, backend, with_schedule=False):
    rzx = linearized_rzx(phi_value, cx_schedule, phi_to_width, backend, with_schedule=with_schedule)
    
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
    
def linearized_rxx