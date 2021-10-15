import collections
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit

from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData, FitVal
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.curve_analysis import curve_fit

class PhaseDependenceAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options()
    
    def _run_analysis(self, experiment_data):
        xdata = experiment_data.metadata['phi_values']
        num_qubits = experiment_data.metadata['num_qubits']
        
        data = experiment_data.data()
        
        zeros = [{'sx': np.zeros_like(xdata), 'x': np.zeros_like(xdata)} for _ in range(num_qubits)]
        ones = [{'sx': np.zeros_like(xdata), 'x': np.zeros_like(xdata)} for _ in range(num_qubits)]
        
        for datum in data:
            iq = datum['metadata']['qubit']
            gate = datum['metadata']['gate']
            iphi = datum['metadata']['phi_index']
            zeros[iq][gate][iphi] = datum['counts'].get('0', 0)
            ones[iq][gate][iphi] = datum['counts'].get('1', 0)
            
        results = []
        plots = []
            
        def fun(x, phi0, amp, offset):
            return offset + amp * np.cos(x + phi0)
        
        p0 = (0., 0.1, 0.5)

        for iq in range(num_qubits):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(r'$\phi$')
            ax.set_ylabel(r'$\langle Z \rangle - \langle Z \rangle_{th}$')
            ax.set_title('Q{}'.format(iq))
            
            ax.axhline(y=0.)
            ax.set_ylim(-0.5, 0.5)
            
            for gate in ['sx', 'x']:
                ydata = (zeros[iq][gate] - ones[iq][gate]) / (zeros[iq][gate] + ones[iq][gate])
        
                fit_result = curve_fit(fun, xdata, ydata, p0)
            
                results.append(AnalysisResultData('amplitude_{}'.format(gate), FitVal(fit_result.popt[1], fit_result.popt_err[1]), device_components=['Q{}'.format(iq)]))
                
                if gate == 'x':
                    ydata += 1.

                ax.scatter(xdata, ydata, label=gate)

            ax.legend()
            plots.append(fig)
        
        return results, plots


class PhaseDependenceExperiment(BaseExperiment):
    __analysis_class__ = PhaseDependenceAnalysis
    
    def __init__(self, qubit_list, phi_values):
        super().__init__(qubit_list)
        
        self.phi_values = phi_values
        
    def _additional_metadata(self):
        return {'phi_values': self.phi_values}
        
    def circuits(self, backend):
        circuits = []
        
        for iq in self.physical_qubits:
            for iphi, phi_value in enumerate(self.phi_values):
                circuit = QuantumCircuit(self.num_qubits, 1)
                circuit.rz(phi_value, iq)
                circuit.sx(iq)
                circuit.measure(iq, 0)
                circuit.name = 'sx'
                circuit.metadata = {'qubit': iq, 'gate': 'sx', 'phi': phi_value, 'phi_index': iphi}
                circuits.append(circuit)

            for iphi, phi_value in enumerate(self.phi_values):
                circuit = QuantumCircuit(self.num_qubits, 1)
                circuit.rz(phi_value, iq)
                circuit.x(iq)
                circuit.measure(iq, 0)
                circuit.name = 'x'
                circuit.metadata = {'qubit': iq, 'gate': 'x', 'phi': phi_value, 'phi_index': iphi}
                circuits.append(circuit)

        return circuits