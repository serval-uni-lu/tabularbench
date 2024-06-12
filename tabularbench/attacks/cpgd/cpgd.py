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
from tabularbench.models.tab_scaler import TabScaler


class CPGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        constraints (Constraints) : The constraint object to be checked successively
        scaler (TabScaler): scaler used to transform the inputs
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        constraints: Constraints,
        scaler: TabScaler,
        model,
        model_objective,
        eps=1.0,
        alpha=0.0,  # unused
        steps=10,
        random_start=True,
        eps_for_division=1e-10,
        fix_equality_constraints_end: bool = True,
        fix_equality_constraints_iter: bool = True,
        adaptive_eps: bool = False,
        eps_margin=0.05,
        seed: int = 0,
        fix_constraints_ijcai=False,
        **kwargs,
    ):
        super().__init__("PGDL2", model)
        self.eps_original = eps
        self.eps = eps - eps * eps_margin
        self.alpha = self.eps_original * 0.4
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

        self.constraints = constraints
        self.scaler = scaler
        self.fix_equality_constraints_end = fix_equality_constraints_end
        self.fix_equality_constraints_iter = fix_equality_constraints_iter
        self.adaptive_eps = adaptive_eps

        self.objective_calculator = None
        self.constraints_executor = None
        self.fix_constraints_ijcai = fix_constraints_ijcai

        if self.constraints.relation_constraints is not None:
            self.objective_calculator = ObjectiveCalculator(
                model_objective,
                constraints=self.constraints,
                thresholds={"distance": eps},
                norm="L2",
                fun_distance_preprocess=self.scaler.transform,
            )
            self.constraints_executor = ConstraintsExecutor(
                AndConstraint(self.constraints.relation_constraints),
                PytorchBackend(),
                feature_names=self.constraints.feature_names,
            )

        self.mutable_mask = scaler.transform_mask(
            torch.tensor(self.constraints.mutable_features, dtype=torch.float)
        ).to(self.device)

        self.seed = seed

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if hasattr(self, "scaler") and self.scaler is not None:
            inputs = self.scaler.inverse_transform(inputs)
        return super().get_logits(inputs, labels, *args, **kwargs)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # self._check_inputs(images)

        x = images
        x_in = images.clone()
        x = self.scaler.transform(x)

        x = x.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv = self.perturb(x, labels)

        x = self.scaler.inverse_transform(x)
        adv = self.scaler.inverse_transform(adv)

        adv = fix_types(x_in, adv, self.constraints.feature_types)
        adv = fix_immutable(x_in, adv, self.constraints.mutable_features)

        if self.fix_equality_constraints_end:
            adv = fix_equality_constraints(
                self.constraints, adv, self.fix_constraints_ijcai
            )

        return adv

    def perturb(self, images, labels):
        r"""
        Overridden.
        """
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps * self.mutable_mask
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for step in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # a  = self.scaler.inverse_transform(adv_images)
            # c = self.constraints_executor.execute(a)
            # g = torch.autograd.grad(
            #     c.sum(), a, retain_graph=False, create_graph=False, allow_unused=True
            # )[0]
            if self.constraints.relation_constraints is not None:
                # TODO check if this is correct
                constraints_cost = self.constraints_executor.execute(
                    self.scaler.inverse_transform(adv_images)
                )
                cost = cost - constraints_cost.sum()

            # Update adversarial images
            grad = torch.autograd.grad(
                cost,
                adv_images,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_norms = (
                torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                + self.eps_for_division
            )  # nopep8
            grad = grad / grad_norms.view(batch_size, 1)

            local_alpha = self.alpha
            if self.adaptive_eps:
                iteration_per_step = self.steps // 7
                current_power = torch.tensor(
                    step // iteration_per_step + 1
                ).float()
                local_alpha = self.eps_original * (
                    1 / torch.float_power(10.0, current_power)
                )
                print(local_alpha)

            adv_images = (
                adv_images.detach() + local_alpha * grad * self.mutable_mask
            )

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)

            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(
                -1,
                1,
            )

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            if self.fix_equality_constraints_iter:
                adv_images = self.scaler.transform(
                    fix_equality_constraints(
                        self.constraints,
                        self.scaler.inverse_transform(adv_images),
                        self.fix_constraints_ijcai,
                    )
                )

            # Fixes

        return adv_images
