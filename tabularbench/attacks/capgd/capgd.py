import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchattacks.attack import Attack

from tabularbench.attacks.objective_calculator import ObjectiveCalculator
from tabularbench.attacks.utils import (
    fix_equality_constraints,
    fix_immutable,
    fix_types,
)
from tabularbench.constraints.constraints import Constraints
from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.models.model import Model
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.datatypes import to_numpy_number


class CAPGD(Attack):
    r"""
    CAPGD in the paper 'Towards Adaptive Attacks on Constrained Tabular Machine Learning'
    [https://openreview.net/forum?id=DnvYdmR9OB]
    

    Distance Measure : Linf, L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (tabularbench.models.model): model to attack.
        model_objective (tabularbench.models.model): model used to compute the objective.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        fix_equality_constraints_end (bool): whether to fix equality constraints at the end. (Default: True)
        fix_equality_constraints_iter (bool): whether to fix equality constraints at each iteration. (Default: True)
        adaptive_eps (bool): whether to use adaptive epsilon. (Default: True)
        random_start (bool): whether to use random start. (Default: True)
        init_start (bool): whether to initialize the starting point. (Default: True)
        best_restart (bool): whether to use the best restart. (Default: True)
        eps_margin (float): margin for epsilon. (Default: 0.05)
        verbose (bool): print progress. (Default: False)

    Shape:
        - inputs: (N, D)
        - labels: (N, C)

    Returns:
        - outputs: (N, C)

    Examples::
        >>> attack = CAPGD(...)
        >>> outputs = attack(inputs, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        scaler: TabScaler,
        model: Model,
        model_objective: Model,
        norm: str = "Linf",
        eps: float = 8 / 255,
        steps: int = 10,
        n_restarts: int = 1,
        seed: int = 0,
        loss: str = "ce",
        eot_iter: int = 1,
        rho: float = 0.75,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        adaptive_eps: bool = True,
        random_start: bool = True,
        init_start: bool = True,
        best_restart: bool = True,
        eps_margin: float = 0.05,
        verbose: bool = False,
    ) -> None:
        super().__init__("CAPGD", model)
        self.constraints = constraints
        self.scaler = scaler
        self.eps = eps - eps * eps_margin
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.adaptive_eps = adaptive_eps
        self.random_start = random_start
        self.init_start = init_start
        self.best_restart = best_restart

        self.objective_calculator: Optional[ObjectiveCalculator] = None
        self.constraints_executor: Optional[ConstraintsExecutor] = None
        self.objective_calculator = ObjectiveCalculator(
            model_objective,
            constraints=self.constraints,
            thresholds={"distance": eps},
            norm=norm,
            fun_distance_preprocess=self.scaler.transform,
        )
        if self.constraints.relation_constraints is not None:

            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints.feature_names,
            )

        self.mutable_mask = scaler.transform_mask(
            torch.tensor(self.constraints.mutable_features, dtype=torch.float)
        ).to(self.device)

    def forward(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        r"""
        input shape: [N, D]
        output shape: [N, C]
        """
        # self._check_inputs(inputs)

        x = inputs
        x_in = inputs.clone()
        x = self.scaler.transform(x)

        x = x.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        _, adv = self.perturb(x, labels, x_in, cheap=True)

        x = self.scaler.inverse_transform(x)
        adv = self.scaler.inverse_transform(adv)

        adv = fix_types(x_in, adv, self.constraints.feature_types)
        adv = fix_immutable(x_in, adv, self.constraints.mutable_features)

        if self.fix_equality_constraints_end:
            adv = fix_equality_constraints(self.constraints, adv)

        return adv

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in, n_restart):
        x = x_in.clone() if len(x_in.shape) == 2 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ",
                self.steps,
                self.steps_2,
                self.steps_min,
                self.size_decr,
            )

        # -- Random initialization
        random_start = (self.n_restarts != 1) and (n_restart != 0)
        random_start = random_start or (not self.init_start)
        random_start = random_start and (self.random_start)

        if random_start:
            # print("Random")
            if self.norm == "Linf":
                t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
                x_adv = x.detach() + self.mutable_mask.to(self.device) * (
                    self.eps
                    * torch.ones(
                        [
                            x.shape[0],
                            1,
                        ]
                    )
                    .to(self.device)
                    .detach()
                    * t
                    / (
                        t.reshape([t.shape[0], -1])
                        .abs()
                        .max(dim=1, keepdim=True)[0]
                        .reshape(
                            [
                                -1,
                                1,
                            ]
                        )
                    )
                )
            elif self.norm == "L2":
                t = torch.randn(x.shape).to(self.device).detach()
                if self.mutable_mask.shape[0] < x.shape[1]:
                    self.mutable_mask = torch.cat(
                        (
                            self.mutable_mask,
                            torch.ones(
                                1,
                            ).to(self.mutable_mask.device),
                        )
                    )

                x_adv = x.detach() + self.mutable_mask * (
                    self.eps
                    * torch.ones(
                        [
                            x.shape[0],
                            1,
                        ]
                    )
                    .to(self.device)
                    .detach()
                    * t
                    / (
                        (t**2)
                        .sum(dim=list(range(1, len(x.shape))), keepdim=True)
                        .sqrt()
                        + 1e-12
                    )
                )
        else:
            # print("Init")
            x_adv = x.detach()
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr":
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError("unknown loss")

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.get_logits(
                    x_adv
                )  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                if self.constraints.relation_constraints is not None:
                    loss_indiv = (
                        loss_indiv
                        - self.constraints_executor.execute(
                            self.scaler.inverse_transform(x_adv)
                        )
                    )

                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[
                0
            ].detach()  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones(
                [
                    x.shape[0],
                    1,
                ]
            )
            .to(self.device)
            .detach()
            * torch.Tensor([2.0])
            .to(self.device)
            .detach()
            .reshape(
                [
                    1,
                    1,
                ]
            )
        )
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(
            loss_best.shape
        )
        n_reduced = 0

        for i in range(self.steps):
            # -- gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + self.mutable_mask.to(x_adv.device) * (
                        step_size * torch.sign(grad)
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(x_adv_1, x - self.eps), x + self.eps
                        ),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv
                                + (x_adv_1 - x_adv) * a
                                + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + self.mutable_mask * (
                        step_size
                        * grad
                        / (
                            (grad**2 * self.mutable_mask)
                            .sum(
                                dim=list(range(1, len(x.shape))), keepdim=True
                            )
                            .sqrt()
                            + 1e-12
                        )
                    )
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2)
                            .sum(
                                dim=list(range(1, len(x.shape))), keepdim=True
                            )
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(
                                dim=list(range(1, len(x.shape))), keepdim=True
                            )
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = (
                        x_adv
                        + self.mutable_mask * (x_adv_1 - x_adv) * a
                        + self.mutable_mask * grad2 * (1 - a)
                    )
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2)
                            .sum(
                                dim=list(range(1, len(x.shape))), keepdim=True
                            )
                            .sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps
                            * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(
                                dim=list(range(1, len(x.shape))), keepdim=True
                            )
                            .sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )

                x_adv = x_adv_1 + 0.0

            if self.fix_equality_constraints_iter:
                # print(f"fixing equality constraints {x_best_adv.shape}")
                x_adv = self.scaler.transform(
                    fix_equality_constraints(
                        self.constraints,
                        self.scaler.inverse_transform(x_adv),
                    )
                )
            # -- get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.get_logits(
                        x_adv
                    )  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    if self.constraints.relation_constraints is not None:
                        loss_indiv = (
                            loss_indiv
                            - self.constraints_executor.execute(
                                self.scaler.inverse_transform(x_adv)
                            )
                        )
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[
                    0
                ].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )

            # x_best_adv = self.scaler.transform(
            #     fix_equality_constraints(
            #         self.constraints,
            #         self.scaler.inverse_transform(x_best_adv),
            #     )
            # )
            if self.verbose:
                print(
                    "iteration: {} - Best loss: {:.6f}".format(
                        i, loss_best.sum()
                    )
                )

            # -- check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if (counter3 == k) and self.adaptive_eps:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy()
                        >= loss_best.cpu().numpy()
                    )
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        if self.best_restart:
                            x_adv[fl_oscillation] = x_best[
                                fl_oscillation
                            ].clone()
                            grad[fl_oscillation] = grad_best[
                                fl_oscillation
                            ].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        # We use x_best_adv as adversarial
        return x_best, acc, loss_best, x_best_adv

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if hasattr(self, "scaler") and self.scaler is not None:
            inputs = self.scaler.inverse_transform(inputs)
        return super().get_logits(inputs, labels, *args, **kwargs)

    def perturb(
        self,
        x_in,
        y_in,
        x_in_unscaled,
        best_loss=False,
        cheap=True,
    ):
        assert self.norm in ["Linf", "L2"]
        x = x_in.clone() if len(x_in.shape) == 2 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.get_logits(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError("not implemented yet")

            else:
                for counter in range(self.n_restarts):
                    # ind_to_fool = acc.nonzero().squeeze()

                    # if self.objective_calculator is not None:
                    #     x_clean = self.scaler.inverse_transform(x_in.clone().cpu())
                    #     x_adv_for_ind = fix_types(
                    #         x_clean=x_clean,
                    #         x_adv=self.scaler.inverse_transform(adv.cpu()),
                    #         types=self.constraints.feature_types,
                    #     )

                    #     ind_to_fool = self.objective_calculator.get_unsuccessful_attacks_clean_indexes(
                    #         x_clean=x_clean.cpu().numpy(),
                    #         y_clean=y_in.cpu().numpy(),
                    #         x_adv=x_adv_for_ind.clone()
                    #         .detach()
                    #         .numpy()[:, np.newaxis, :],
                    #     )
                    if counter == 0:
                        ind_to_fool = np.arange(len(x))
                    ind_to_fool = (
                        torch.from_numpy(ind_to_fool).to(self.device).long()
                    )
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        # print(f"IDX: Fool {len(ind_to_fool)}")
                        x_to_fool, y_to_fool = (
                            x[ind_to_fool].clone(),
                            y[ind_to_fool].clone(),
                        )
                        (
                            best_curr,
                            acc_curr,
                            loss_curr,
                            adv_curr,
                        ) = self.attack_single_run(
                            x_to_fool, y_to_fool, counter
                        )
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        adv[ind_to_fool] = adv_curr.clone()
                        adv1 = adv_curr.clone()
                        adv1 = self.scaler.inverse_transform(adv_curr)
                        adv1 = fix_types(
                            x_in_unscaled[ind_to_fool],
                            adv1,
                            self.constraints.feature_types,
                        )
                        adv1 = fix_immutable(
                            x_in_unscaled[ind_to_fool],
                            adv1,
                            self.constraints.mutable_features,
                        )

                        adv1 = fix_equality_constraints(self.constraints, adv1)
                        adv1 = to_numpy_number(adv1)[:, np.newaxis, :]
                        x_clean1 = to_numpy_number(
                            x_in_unscaled[ind_to_fool]
                        ).astype(np.float32)
                        (
                            success_attack_indices,
                            success_adversarials_indices,
                        ) = self.objective_calculator.get_successful_attacks_indexes(
                            x_clean1, y[ind_to_fool].cpu(), adv1, max_inputs=1
                        )
                        # print(f"IDX: Success {len(success_attack_indices)}")
                        adv[ind_to_fool[success_attack_indices]] = (
                            self.scaler.transform(torch.tensor(adv1[:, 0, :]))[
                                success_attack_indices
                            ]
                        )
                        ind_to_fool = np.setdiff1d(
                            ind_to_fool, ind_to_fool[success_attack_indices]
                        )
                        if self.verbose:
                            print(
                                "restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s".format(
                                    counter,
                                    acc.float().mean(),
                                    time.time() - startt,
                                )
                            )

            return acc, adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (
                -float("inf")
            )
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print(
                        "restart {} - loss: {:.5f}".format(
                            counter, loss_best.sum()
                        )
                    )

            return loss_best, adv_best

# __all__ definition
__all__ = ['CAPGD']