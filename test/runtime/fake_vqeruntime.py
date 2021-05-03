# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fake runtime provider and VQE runtime."""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import Provider


class FakeJob():
    """A fake job for unittests."""

    def result(self) -> None:
        """Return nothing."""
        return None


class FakeVQERuntime():
    """A fake VQE runtime for unittests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != 'vqe':
            raise ValueError('program_id is not vqe.')

        allowed_inputs = {
            'operator': PauliSumOp,
            'aux_operators': (list, type(None)),
            'ansatz': QuantumCircuit,
            'initial_point': (np.ndarray, str),
            'optimizer': str,
            'optimizer_params': dict,
            'shots': int,
            'readout_error_mitigation': bool
        }
        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f'{arg} does not have the right type: {allowed_inputs[arg]}')

        allowed_options = {
            'backend_name': str
        }
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f'{arg} does not have the right type: {allowed_inputs[arg]}')

        if callback is not None:
            try:
                fake_job_id = 'c919jdjlwinoir1a'
                fake_data = [3, np.arange(10), 1.3]
                _ = callback(fake_job_id, fake_data)
            except Exception as exc:
                raise ValueError('Callback failed') from exc

        return FakeJob()


class FakeRuntimeProvider(Provider):
    """A fake runtime provider for unittests."""

    def has_service(self, service):
        """Check if a service is available."""
        if service == 'runtime':
            return True
        return False

    @property
    def runtime(self) -> FakeVQERuntime:
        """Return the runtime."""
        return FakeVQERuntime()
