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

"""Test the VQE program."""

from test.nature_test_case import QiskitNatureTestCase

import numpy as np
from qiskit.providers.aer import AerSimulator
from qiskit_nature.runtime import VQEProgram
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import X, I, Z
from .fake_vqeruntime import FakeRuntimeProvider


class TestVQEProgram(QiskitNatureTestCase):
    """Test the VQE program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeRuntimeProvider()

    def test_standard_case(self):
        """Test a standard use case."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.random(circuit.num_parameters)
        optimizer_name = 'SPSA'
        optimizer_settings = {'maxiter': 100}
        backend = AerSimulator()

        vqe = VQEProgram(ansatz=circuit,
                         optimizer_name=optimizer_name,
                         optimizer_settings=optimizer_settings,
                         initial_point=initial_point,
                         backend=backend,
                         provider=self.provider
                         )
        result = vqe.compute_minimum_eigenvalue(operator)
