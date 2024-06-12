import numpy as np
from pymoo.core.sampling import Sampling


class InitialStateSampling(Sampling):
    """
    Repeat the initial state
    """

    def __init__(self, type_mask) -> None:
        self.type_mask = type_mask
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Retrieve original
        x_clean = problem.x_clean[problem.constraints.mutable_features]

        x_generated = np.tile(x_clean, (n_samples, 1))

        mask_int = self.type_mask != "real"

        x_generated[:, mask_int] = np.rint(x_generated[:, mask_int]).astype(
            int
        )

        return x_generated
