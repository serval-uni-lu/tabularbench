import time
from warnings import warn

import numpy as np
import torch
from torchattacks.attack import Attack
from torchattacks.wrappers.multiattack import MultiAttack

from tabularbench.attacks.capgd.capgd import CAPGD
from tabularbench.attacks.moeva.moeva import Moeva2
from tabularbench.attacks.objective_calculator import ObjectiveCalculator
from tabularbench.constraints.constraints import Constraints
from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.numpy_backend import NumpyBackend
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.datatypes import to_numpy_number, to_torch_number


class NoAttack:
    """Utility class to have no attack."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        N: Number of instances
        D: Number of features
        C: Number of classes
        input shape: [N, D]
        output shape: [N, C]
        """
        return x


class ConstrainedMultiAttack(MultiAttack):
    r"""
    Constrained Multi Attack (CMA)
    A generic class to run multiple attacks iteratively while checking the constraint and success rate at each step, and running the nextr attacks only on unsuccessful examples

    Arguments:
        objective_calculator (ObjectiveCalculator): The objective calculator to be used.
        attacks (list): List of attacks to be used.
        verbose (bool): Whether to print the progress. (Default: False)
    """
    def __init__(self, objective_calculator, attacks, verbose=False):
        super(ConstrainedMultiAttack, self).__init__(attacks, verbose=verbose)
        self.objective_calculator = objective_calculator
        self.attack_times = []
        self.robust_accuracies = []
        self.constraints_rate = []
        self.distance_ok = []
        self.mdc = []

    # Override from upper class
    # Moeva does not use the same model as other attacks
    def check_validity(self):
        r"""
        Check if at least 2 attacks are available, if the model used in the attack is compatible with each and is the same.
        """
        if len(self.attacks) < 2:
            warn("More than two attacks should be given.")

        ids = []
        for attack in self.attacks:
            if isinstance(attack, Moeva2):
                if hasattr(attack.model, "__self__"):
                    ids.append(id(attack.model.__self__.wrapper_model))
                else:
                    ids.append(id(attack.model.wrapper_model))
            else:
                ids.append(id(attack.model))
        if len(set(ids)) != 1:
            raise ValueError(
                "At least one of attacks is referencing a different model."
            )

    def forward(self, inputs, labels):
        r"""
        N: Number of instances
        D: Number of features
        C: Number of classes
        input shape: [N, D]
        output shape: [N, C]
        """
        batch_size = inputs.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        def constraints_distance(c, x):
            x_dim = len(x.shape)
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])

            ce = ConstraintsExecutor(
                c,
                backend=NumpyBackend(),
                feature_names=self.objective_calculator.constraints.feature_names,
            )
            distance = ce.execute(x)
            distance = distance.reshape(x_shape[: x_dim - 1])
            return distance

        def constraints_percentage(c, x, measure):
            x_dim = len(x.shape)
            distance = constraints_distance(c, x)

            constraint_ok = (
                distance <= self.objective_calculator.thresholds["constraints"]
            )

            if x_dim == 2:
                constraint_ok = constraint_ok[:, np.newaxis, :]

            percentage = constraint_ok
            self.distance_ok.append(np.mean(measure.distance))

            percentage = np.mean(percentage, axis=1)
            percentage = np.mean(percentage, axis=0)

            return percentage

        for attack_i, attack in enumerate([NoAttack()] + self.attacks):
            # Attack the one that failed so far
            # if attack_i > 0:
            #     exit(0)
            start_time = time.time()
            # attack.objective_calculator = self.objective_calculator
            adv_inputs = attack(inputs[fails], labels[fails])
            self.attack_times.append(time.time() - start_time)

            # Correct shape
            filter_adv = (
                adv_inputs.unsqueeze(1)
                if len(adv_inputs.shape) < 3
                else adv_inputs
            )
            # Type conversion
            numpy_clean = to_numpy_number(inputs[fails]).astype(np.float32)
            numpy_adv = to_numpy_number(filter_adv)
            # np.save("b.npy", numpy_adv)
            # np.save("b_clean.npy", numpy_clean)

            # Indexes of the successful attacks
            (
                success_attack_indices,
                success_adversarials_indices,
            ) = self.objective_calculator.get_successful_attacks_indexes(
                numpy_clean, labels[fails].cpu(), numpy_adv, max_inputs=1
            )

            # print(f"Objective_calc id in CAA {id(self.objective_calculator)}")

            # print(f"Index len{len(success_attack_indices)}")
            # print(self.objective_calculator.thresholds)

            # Sanity check start, can ignore for debugging
            # clean_indices = (
            #     self.objective_calculator.get_successful_attacks_clean_indexes(
            #         numpy_clean, labels[fails].cpu(), numpy_adv, recompute=False
            #     )
            # )
            # assert np.equal(clean_indices, success_attack_indices).all()
            # Sanity check end
            # print("After check")

            # If we found adversarials
            if len(success_attack_indices) > 0:
                final_inputs[fails[success_attack_indices]] = filter_adv[
                    success_attack_indices, success_adversarials_indices, :
                ].squeeze(1)
                mask = torch.ones_like(fails)
                mask[success_attack_indices] = 0
                fails = fails.masked_select(mask.bool())

            multi_atk_records.append(len(fails))

            self.robust_accuracies.append(1 - (len(fails) / batch_size))

            # measure = self.objective_calculator.get_objectives_respected(
            #     numpy_clean, labels[fails].cpu(), numpy_adv, recompute=False
            # )
            # self.constraints_rate.append(
            #     [
            #         constraints_percentage(c, numpy_adv, measure)
            #         for c in self.objective_calculator.constraints.relation_constraints
            #     ]
            # )

            self.mdc.append(
                self.objective_calculator.get_success_rate(
                    numpy_clean,
                    labels[fails].cpu(),
                    numpy_adv,
                    recompute=False,
                )
            )

            if len(fails) == 0:
                break

        if self.verbose:
            print(self._return_sr_record(multi_atk_records))

        if self._accumulate_multi_atk_records:
            self._update_multi_atk_records(multi_atk_records)

        return final_inputs


class ConstrainedAutoAttack(Attack):
    r"""
    CAA from "Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data"
    [https://arxiv.org/abs/2406.00775]

    License: MIT
    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be used in the attack
        constraints_eval (Constraints) : The constraint object to be checked at the end
        scaler (TabScaler): scaler used to transform the inputs
        model (tabularbench.models.model): model to attack.
        model_objective (tabularbench.models.model): model used to compute the objective.
        n_jobs (int): number of parallel jobs. (Default: -1)
        fix_equality_constraints_end (bool): whether to fix equality constraints at the end. (Default: True)
        fix_equality_constraints_iter (bool): whether to fix equality constraints at each iteration. (Default: True)
        eps_margin (float): margin for epsilon. (Default: 0.05)
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (str): version. ['standard'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        steps (int): number of steps. (Default: 10)
        n_gen (int): number of generations. (Default: 100)
        n_offsprings (int): number of offsprings. (Default: 100)

    Shape:
        - inputs: torch.Tensor `(N, F)` where `N = number of batches`, `F=Number of features`.
        - labels: torch.Tensor `(N, C)` where N = number of batches`, `C=number of classes`. (only binary for now)
        - output: torch.Tensor `(N, F)` where `N = number of batches`, `F=Number of features`.

    Examples::
        >>> attack = ConstrainedAutoAttack(...)
        >>> outputs = attack(inputs, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        constraints_eval: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        n_jobs=-1,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        eps_margin=0.01,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
        steps=10,
        n_gen=100,
        n_offsprings=100,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.constraints = constraints
        self.constraints_eval = constraints_eval
        self.scaler = scaler
        self.eps_margin = eps_margin
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.n_jobs = n_jobs
        self.steps = steps
        self.n_gen = n_gen
        self.n_offsprings = n_offsprings

        self.objective_calculator = ObjectiveCalculator(
            model_objective,
            constraints=self.constraints_eval,
            thresholds={"distance": eps},
            norm=norm,
            fun_distance_preprocess=self.scaler.transform,
        )
        if self.constraints_eval.relation_constraints is not None:
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints_eval.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints_eval.feature_names,
            )
        else:
            self.constraints_executor = None

        if version == "standard":  # ['c-apgd-ce', 'c-fab', 'Moeva2']
            self._autoattack = ConstrainedMultiAttack(
                self.objective_calculator,
                [
                    CAPGD(
                        constraints,
                        scaler,
                        model,
                        model_objective,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=2,
                        fix_equality_constraints_end=fix_equality_constraints_end,
                        fix_equality_constraints_iter=fix_equality_constraints_iter,
                        eps_margin=eps_margin,
                        best_restart=False,
                        steps=self.steps,
                    ),
                    Moeva2(
                        model_objective,
                        constraints=constraints,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        fun_distance_preprocess=scaler.transform,
                        n_jobs=n_jobs,
                        n_gen=self.n_gen,
                        n_offsprings=self.n_offsprings,
                    ),
                ],
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, inputs, labels):
        r"""
        input shape: [N, D]
        output shape: [N, C]

        N: Number of instances
        D: Number of features
        C: Number of classes
        """

        is_numpy = isinstance(inputs, np.ndarray)
        inputs = to_torch_number(inputs)
        labels = to_torch_number(labels).long()

        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_inputs = self._autoattack(inputs, labels)

        if is_numpy:
            adv_inputs = to_numpy_number(adv_inputs)

        return adv_inputs

    def get_seed(self):
        r"""
        Return the seed for the random number generatorsed in the attack
        """
        return int(time.time()) if self.seed is None else self.seed
