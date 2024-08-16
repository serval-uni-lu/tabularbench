# Models

## Existing models

So far, our API natively supports: RLN, STG, TabNet, TabTransformer, VIME, and MLP. 
Our implementation is based on Tabsurvey (Borisov et al., 2021}. All models from this framework can be easily adapted to our API.  

### Models description

- TabTransformer is a transformer-based model Huang et al . (2020). It employs self-attention to convert categorical features into an interpretable contextual embedding, which the paper asserts
enhances the model’s robustness to noisy inputs.
- TabNet is another transformer-based model Arik and Pfister (2021). It utilizes multiple sub-networks in sequence. At each decision step, it applies sequential attention to select which features to
consider. TabNet combines the outputs of each step to make the final decision.
- RLN or Regularization Learning Networks Shavitt and Segal (2018) employs an efficient hyperparameter tuning method to minimize counterfactual loss. The authors train a regularization coefficient
for the neural network weights to reduce sensitivity and create very sparse networks.
- STG or Stochastic Gates Yamada et al. (2020) uses stochastic gates for feature selection in neural network estimation tasks. The technique is based on a probabilistic relaxation of the l0 norm of
features or the count of selected features.
- VIME or Value Imputation for Mask Estimation Yoon et al. (2020) employs self-supervised and semi-supervised learning through deep encoders and predictors.
- MLP or Multi Layer Preceptron uses a series of fully connected layers. 

Each dataset requires a set of hyper-parameters to be tuned for optimal performances.

| Family         | Model          | Hyperparameters                                                                                |
|----------------|----------------|------------------------------------------------------------------------------------------------|
| Transformer    | TabTransformer | `hidden_dim`, `n_layers`, `learning_rate`, `norm`, `θ`                                         |
| Transformer    | TabNet         | `n_d`, `n_steps`, `γ`, `cat_emb_dim`, `n_independent`, `n_shared`, `momentum`, `mask_type`     |
| Regularization | RLN            | `hidden_dim`, `depth`, `heads`, `weight_decay`, `learning_rate`, `dropout`                     |
| Regularization | STG            | `hidden_dims`, `learning_rate`, `lam`                                                          |
| Encoding       | VIME           | `p_m`, `α`, `K`, `β`                                                                           |
| Vanilla        | MLP            | `hidden_dim`, `n_layers`                                                                       |


### Using an existing model

Our API supports five architectures, and for each, six data augmentation techniques (as well as no data augmentation) and two training schemes (standard training and adversarial training). Hence, 70 pre-trained models for each of our five datasets are accessible. Below, we fine-tune with CAA AT and CTGAN augmentation a pre-trained Tabtransformer with Cutmix augmentation: 
The method *load_model_and_weights* initialize a pre-trained model (which weights are available in the folder *data/models*) given:
- dataset: the alias of the dataset (url, wids, ...)
- model_arch: the alias of architecture (stg, vime, ...)
- model_training: the type of training:
  - "default": Standard training
  - "cutmix": Online data augmentation with cutmix

```python
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.benchmark.model_utils import load_model_and_weights

dataset = "url"
model_arch = "stg"
model_training= "default"
ds = get_dataset(dataset)
metadata = ds.get_metadata(only_x=True)
scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)

model_eval = load_model_and_weights(
        ds.name, model_arch, model_training, metadata, scaler
    )

```

## Training a new model

### Using an existing architecture and training protocol

```python
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.models.tabsurvey.tabtransformer import TabTransformer
from torch.utils.data import DataLoader


dataset = "url"
ds = get_dataset(dataset)
x, y = ds.get_x_y()
i_train = ds.get_splits()["train"]
x_train, y_train = x.iloc[i_train].to_numpy(), y[i_train]
metadata = ds.get_metadata(only_x=True)

scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
scaler.fit(x_train, metadata["type"])
model = TabTransformer("regression", metadata, scaler=scaler,pretrained="LCLD_TabTr_Cutmix")
train_dataloader = DataLoader(dataset=ds)
model.fit(x=None, y=None, custom_train_dataloader=train_dataloader)
```

### Building a new architecture
All models need to extend the class [BaseModelTorch](https://github.com/serval-uni-lu/tabularbench/blob/main/tabularbench/models/torch_models.py)
. This class implements the definitions, the fit and evaluation methods, and the save and loading methods. Depending of the architectures, scaler and feature encoders can be required by the constructors.

## Submitting a new model

We welcome models contributions that bring additional insights or challenges to the tabular adversarial robustness community.

1. Create [a new issue](https://github.com/serval-uni-lu/tabularbench/issues/new/choose) by selecting the type "Submit a new model".
2. Fill in the form accordingly.
3. Create a new Pull Request with the model definition and files (.py, pre-trained weights .pt, ...) and associate it with this issue.
4. We will validate that the model is working correctly, and run the benchmark on it to confirm the claims.
5. Once included, the model will be accessible on the leaderboard, and if agreed on in the form in the model zoo as well.

If you find issues with existing model in the benchmark or the zoo, please raise a dedicated issue and do not use the form.

Thank you for your contributions.
