from qiskit import QuantumRegister
import numpy as np
from IPython.display import display, Markdown

Markdown('$\newcommand{ket}[1]{|#1\rangle}$')

class StatevectorInspector(object):
    def __init__(self, statevector, registers):
        self._register_size = {}
        self._registers = []
        
        shape = []
        # register list in qiskit goes from least significant and numpy shape is from most significant
        for reg in reversed(registers):
            if type(reg) is tuple: # (name, size)
                shape.append(2 ** reg[1])
                self._register_size[reg[0]] = reg[1]
                self._registers.append(reg[0])
            elif isinstance(reg, QuantumRegister):
                shape.append(2 ** reg.size)
                self._register_size[reg.name] = reg.size
                self._registers.append(reg.name)
            else:
                raise NotImplementedError('Unhandled input type ' + str(type(reg)))

        state = statevector.reshape(tuple(shape))
        state.real = np.where(np.abs(state.real) > 1.e-8, state.real, 0.)
        state.imag = np.where(np.abs(state.imag) > 1.e-8, state.imag, 0.)
        self.state = state
        self.probs = np.square(np.abs(state))

    def get_subsystem(self, subsystem, nonzero_only=True, squeeze=True):
        """Get the unnormalized statevector projected to the subsystem.

        Args:
            subsystem (dict): {register: indices}
            nonzero_only (bool): Return vector elements with nonzero amplitude only.
            squeeze (bool): Squeeze out registers with length 1.

        Returns:
            dict: {config: amplitude} Subsystem statevector.
        """

        flat_array_length = 1
        index_arrays = []
        registers = []
        used_dims = []

        for ireg, reg in enumerate(self._registers):
            if reg in subsystem:
                idx = subsystem[reg]
                if isinstance(idx, int):
                    idx = [idx]
            else:
                idx = np.arange(2 ** self._register_size[reg])

            arr = np.array(idx)

            if not squeeze or arr.shape[0] != 1:
                used_dims.append(ireg)
                registers.append(reg)                

            for iup, upper in enumerate(index_arrays):
                index_arrays[iup] = np.repeat(upper, arr.shape[0])
            index_arrays.append(np.tile(arr, flat_array_length))
            flat_array_length *= arr.shape[0]

        indices_flat = np.stack(index_arrays, axis=-1)
        state = self.state[tuple(index_arrays)]

        if nonzero_only:
            nonzero_idx = state.nonzero()
            state = state[nonzero_idx]
            indices_flat = indices_flat[nonzero_idx]

        out_state = {}
        for idx, amp in zip(indices_flat, state):
            if squeeze:
                idx = idx[np.array(used_dims)]
                    
            out_state[tuple(idx)] = amp

        return registers, out_state


    def print_state(self, factor_by=None, pi_multiple=False):
        pass

    def get_probs(self, indices=None, sum_over=None, nonzero_only=True, squeeze=True):
        """Compute the probabilities of the specified indices.

        Args:
            indices (dict): {register: indices} Use np.newaxis to list out the register. All
                registers not appearing in the dict will be integrated out.
            sum_over (list[str]): Specify registers to sum over instead of indices to keep.
            nonzero_only (bool): Remove entries with null probability.
            squeeze (bool): Squeeze out registers with length 1.

        Returns:
            list: [register] List of the name of registers
            dict: {index: prob} Index is given as a tuple of integers corresponding to the
                values of the registers in the order given in the first return value.
        """

        flat_array_length = 1
        registers = []
        index_arrays = []
        sum_over_axes = []
        used_dims = []

        if sum_over is None:
            assert(indices is not None)

            for ireg, reg in enumerate(self._registers):
                if reg in indices:
                    idx = indices[reg]
                    if idx is np.newaxis:
                        idx = np.arange(2 ** self._register_size[reg])
                    elif isinstance(idx, int):
                        idx = [idx]

                    arr = np.array(idx)

                    if not squeeze or arr.shape[0] != 1:
                        used_dims.append(ireg - len(sum_over_axes))
                        registers.append(reg)                

                    for iup, upper in enumerate(index_arrays):
                        index_arrays[iup] = np.repeat(upper, arr.shape[0])
                    index_arrays.append(np.tile(arr, flat_array_length))
                    flat_array_length *= arr.shape[0]
                else:
                    sum_over_axes.append(ireg)

        else:
            assert(indices is None)

            for ireg, reg in enumerate(self._registers):
                if reg in sum_over:
                    sum_over_axes.append(ireg)
                else:
                    used_dims.append(ireg - len(sum_over_axes))
                    registers.append(reg)

                    arr = np.arange(2 ** self._register_size[reg])
                    for iup, upper in enumerate(index_arrays):
                        index_arrays[iup] = np.repeat(upper, arr.shape[0])
                    index_arrays.append(np.tile(arr, flat_array_length))
                    flat_array_length *= arr.shape[0]

        indices_flat = np.stack(index_arrays, axis=-1)

        probs = self.probs
        if len(sum_over_axes) != 0:
            probs = np.sum(probs, axis=tuple(sum_over_axes))
        probs = probs[tuple(index_arrays)]

        if nonzero_only:
            nonzero_idx = probs.nonzero()
            probs = probs[nonzero_idx]
            indices_flat = indices_flat[nonzero_idx]

        out_probs = {}
        for idx, prob in zip(indices_flat, probs):
            if squeeze:
                idx = idx[np.array(used_dims)]
                
            out_probs[tuple(idx)] = prob

        return registers, out_probs
