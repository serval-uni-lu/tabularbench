from typing import List, Tuple, Type

from tabularbench.models.model import Model
from tabularbench.models.tabsurvey.mlp_rln import models as mlp_rln_models
from tabularbench.models.tabsurvey.stg import models as stg_models
from tabularbench.models.tabsurvey.tabnet import models as tabnet_models
from tabularbench.models.tabsurvey.tabtransformer import (
    models as tabtransformer_models,
)
from tabularbench.models.tabsurvey.vime import models as vime_models

models: List[Tuple[str, Type[Model]]] = (
    tabtransformer_models
    + mlp_rln_models
    + vime_models
    + tabnet_models
    + stg_models
)
