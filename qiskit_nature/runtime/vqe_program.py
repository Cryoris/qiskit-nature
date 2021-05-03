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

"""The Qiskit Nature VQE Quantum Program."""


from typing import List, Union, Callable, Optional, Any, Dict
import numpy as np

from qiskit import IBMQ
from qiskit import QuantumCircuit
from qiskit.providers import Provider
from qiskit.providers.ibmq.exceptions import IBMQNotAuthorizedError
from qiskit.providers.backend import Backend
from qiskit.algorithms import MinimumEigensolver
from qiskit.algorithms import MinimumEigensolverResult
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.quantum_info import SparsePauliOp


class VQEProgram(MinimumEigensolver):
    """The Qiskit Nature VQE Quantum Program to call the VQE runtime as a MinimumEigensolver."""

    def __init__(self,
                 ansatz: QuantumCircuit,
                 optimizer_name: str = 'SPSA',
                 optimizer_settings: Optional[Dict[str, Any]] = None,
                 initial_point: Optional[Union[List, np.ndarray]] = None,
                 provider: Optional[Provider] = None,
                 backend: Optional[Backend] = None,
                 shots: int = 1024,
                 readout_error_mitigation: bool = True,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None
                 ) -> None:
        """
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer_name: A classical optimizer. Currently only ``'SPSA'`` and ``'QN-SPSA'`` are
                supported. Per default, SPSA is used.
            optimizer_settings: The settings for the classical optimizer.
            backend: The backend to run the circuits on.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            shots: The number of shots to be used
            readout_error_mitigation: Whether or not to use readout error mitigation.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.`
        """
        if optimizer_settings is None:
            optimizer_settings = {}

        # define program name
        self._program_name = 'vqe'

        # store settings
        self._provider = None
        self._ansatz = ansatz
        self._optimizer_name = None
        self._optimizer_settings = optimizer_settings
        self._backend = backend
        self._initial_point = initial_point
        self._shots = shots
        self._readout_error_mitigation = readout_error_mitigation
        self._callback = callback

        # use setter to check for valid inputs
        if provider is not None:
            self.provider = provider

        self.optimizer_name = optimizer_name

    @property
    def provider(self) -> Optional[Provider]:
        """Return the provider."""
        return self._provider

    @provider.setter
    def provider(self, provider: Provider) -> None:
        """Set the provider. Must be a provider that supports the runtime feature."""
        try:
            _ = hasattr(provider, 'runtime')
        except IBMQNotAuthorizedError:
            # pylint: disable=raise-missing-from
            raise ValueError(f'The provider {provider} does not provide a runtime environment.')

        self._provider = provider

    @property
    def program_name(self) -> str:
        """Return the program name."""
        return self._program_name

    @property
    def ansatz(self) -> QuantumCircuit:
        """Return the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit) -> None:
        """Set the ansatz."""
        self._ansatz = ansatz

    @property
    def optimizer_name(self) -> str:
        """Return the name of the optimizer."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, name: str) -> None:
        """Return the name of the optimizer."""
        if name not in ['SPSA', 'QN-SPSA']:
            raise NotImplementedError('Only SPSA and QN-SPSA are currently supported.')
        self._optimizer_name = name

    @property
    def optimizer_settings(self) -> Dict[str, Any]:
        """Return the settings of the optimizer."""
        return self._optimizer_settings

    @optimizer_settings.setter
    def optimizer_settings(self, settings: Dict[str, Any]) -> None:
        """Return the settings of the optimizer."""
        self._optimizer_settings = settings

    @optimizer_settings.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """ Sets the ansatz. """
        self._optimizer = optimizer

    @property
    def backend(self) -> Optional[Backend]:
        """Returns the backend."""
        return self._backend

    @backend.setter
    def backend(self, backend) -> None:
        """Sets the backend."""
        self._backend = backend

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets the initial point."""
        self._initial_point = initial_point

    @property
    def shots(self) -> int:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, shots: int) -> None:
        """Set the number of shots."""
        self._shots = shots

    @property
    def readout_error_mitigation(self) -> bool:
        """Returns whether or not to use readout error mitigation.

        Readout error mitigation is done using a complete measurement fitter with the
        ``self.shots`` number of shots and re-calibrations every 30 minutes.
        """
        return self._readout_error_mitigation

    @readout_error_mitigation.setter
    def readout_error_mitigation(self, readout_error_mitigation: bool) -> None:
        """Whether or not to use readout error mitigation. """
        self._readout_error_mitigation = readout_error_mitigation

    @property
    def callback(self) -> Callable:
        """Returns the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable) -> None:
        """Set the callback."""
        self._callback = callback

    def _wrap_vqe_callback(self) -> Optional[Callable]:
        """ Wraps and returns the given callback to match the signature of the runtime callback. """

        def wrapped_callback(*args):
            # TODO: need to be completed
            _, data = args  # first element is the job id
            iteration_count = data[0]
            params = data[1]
            mean = data[2]
            # sigma = 0.0  # TODO
            accepted = data[4]
            metric = data[5]
            return self._callback(iteration_count, params, mean, accepted, metric)

        # if callback is set, return wrapped callback, else return None
        if self._callback:
            return wrapped_callback
        else:
            return None

    def compute_minimum_eigenvalue(self,
                                   operator: OperatorBase,
                                   aux_operators: Optional[List[Optional[OperatorBase]]] = None
                                   ) -> MinimumEigensolverResult:
        """Calls the VQE Runtime to approximate the ground state of the given operator.

        Args:
            operator: Qubit operator of the observable
            aux_operators: Optional list of auxiliary operators to be evaluated with the
                (approximate) eigenstate of the minimum eigenvalue main result and their expectation
                values returned. For instance in chemistry these can be dipole operators, total
                particle count operators so we can get values for these at the ground state.

        Returns:
            MinimumEigensolverResult

        Raises:
            ValueError: If the backend has not yet been set.

        """
        if self.backend is None:
            raise ValueError('The backend has not been set.')

        if self.provider is None:
            raise ValueError('The provider has not been set.')

        if not isinstance(operator, PauliSumOp):
            try:
                primitive = SparsePauliOp(operator.primitive)
                operator = PauliSumOp(primitive, operator.coeff)
            except Exception as exc:
                raise ValueError(f'Invalid type of the operator {type(operator)} '
                                 'must be PauliSumOp, or castable to one.') from exc

        _validate_optimizer_settings(self.optimizer_name, self.optimizer_settings)

        if self.initial_point is None:
            initial_point = 'random'
        else:
            initial_point = self.initial_point

        # combine the settings with the given operator to runtime inputs
        # TODO: change 'random' to None in runtime inputs for initial state
        inputs = {
            'operator': operator,
            'aux_operators': aux_operators,
            'ansatz': self.ansatz,
            'optimizer': self.optimizer_name,
            'optimizer_params': self.optimizer_settings,
            'initial_point': initial_point,
            'shots': self.shots,
            'readout_error_mitigation': self.readout_error_mitigation
        }

        # define runtime options
        options = {
            'backend_name': self.backend.name()
        }

        # send job to runtime and return result
        return self.provider.runtime.run(program_id=self.program_name,
                                         inputs=inputs,
                                         options=options,
                                         callback=self._wrap_vqe_callback()
                                         ).result()


def _validate_optimizer_settings(name, settings):
    if name not in ['SPSA', 'QN-SPSA']:
        raise NotImplementedError('Only SPSA and QN-SPSA are currently supported.')

    allowed_settings = [
        'maxiter',
        'blocking',
        'allowed_increase',
        'trust_region',
        'learning_rate',
        'perturbation',
        'resamplings',
        'last_avg',
        'second_order',
        'hessian_delay',
        'regularization',
        'initial_hessian'
    ]

    if name == 'QN-SPSA':
        allowed_settings.remove(['trust_region', 'second_order'])

    unsupported_args = set(settings.keys()) - set(allowed_settings)

    if len(unsupported_args) > 0:
        raise ValueError(f'The following settings are unsupported for the {name} optimizer: '
                         f'{unsupported_args}')
