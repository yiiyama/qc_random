import math
import numpy as np
import scipy.special as scispec
from qiskit.transpiler.passes.calibration.builders import *
from qiskit.pulse import Gaussian

class RZXCalibrationBuilderWithPedestal(RZXCalibrationBuilderNoEcho):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate.
    The ``RZXCalibrationBuilderNoEcho`` is a variation of the
    :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder` pass
    that creates calibrations for the cross-resonance pulses without inserting
    the echo pulses in the pulse schedule. This enables exposing the echo in
    the cross-resonance sequence as gates so that the transpiler can simplify them.
    The ``RZXCalibrationBuilderNoEcho`` only supports the hardware-native direction
    of the CX gate.
    """
    
    @staticmethod
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> Play:
        """
        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.
        Returns:
            qiskit.pulse.Play: The play instruction with the stretched compressed
                GaussianSquare pulse.
        Raises:
            QiskitError: if the pulses are not GaussianSquare.
        """
        pulse_ = instruction.pulse
        if isinstance(pulse_, GaussianSquare):
            amp = pulse_.amp
            width = pulse_.width
            sigma = pulse_.sigma
            duration = pulse_.duration

            sign = theta / abs(theta)
            
            flank_width = (duration - width) // 2

            normal_flank_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(flank_width / np.sqrt(2.) / sigma)
            pedestal = np.exp(-0.5 * np.square(flank_width / sigma))
            grounded_flank_integral = (normal_flank_integral - pedestal * flank_width) / (1. - pedestal)
            flank_area = np.abs(amp) * grounded_flank_integral
            angle_to_area = (np.abs(amp) * width + 2. * flank_area) / (np.pi / 2.)

            minimum_theta = 2. * flank_area / angle_to_area
            
            if abs(theta) > minimum_theta:
                target_area = abs(theta) * angle_to_area
                width = (target_area - 2. * flank_area) / np.abs(amp)
                duration = math.ceil((width + 2 * flank_width) / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(amp=sign * amp, width=width, sigma=sigma, duration=duration),
                    channel=instruction.channel,
                )                
            else:
                amp_scale = theta / minimum_theta
                duration = math.ceil(2 * flank_width / sample_mult) * sample_mult
                return Play(
                    Gaussian(amp=amp * amp_scale, sigma=sigma, duration=duration),
                    channel=instruction.channel,
                )
        else:
            raise QiskitError("RZXCalibrationBuilder only stretches/compresses GaussianSquare.")
