# Constraints

One of the features of our benchmark is the support of feature constraints, in the dataset definition and in the attacks. 

There three 3 types of constraints:

1. Boundary constraints: These constraints are defined in meta_data of dataset (for example the csv) and define the maximum and minimum allowed values for each feature.
2. Mutability constraints: These constraints are also defined in meta_data of dataset (for example the csv) and indicate which features can 

## Manual definition of feature relations

all classes below are defined in **tabularbench.constraints.relation_constraint**.

Constraints between features can be expressed in natural language. For example, we express the constraint F0 = F1 + F2 such as:
```python
from tabularbench.constraints.relation_constraint import Feature
constraint1 = Feature(0) == Feature(1) + Feature(2)
```

### Accessing a feature

A feature can be accessed by its index (0, 1, ...) or by its name (*installment*,*loan_amnt*, ...).

```python
from tabularbench.constraints.relation_constraint import Feature
constraint1 = Feature(0) == Feature(1) + Feature(2)
constraint2 = Feature("open_acc") <= Feature("total_acc")
```

### Pre-builts

- Base operators: Pre-built operators include equalities, inequalities, math operations (+,-,*,/, ...) custom operators can be built by extending the class *MathOperation*:
- Safe operators: *SafeDivision* and *Log* allow a fallback value if the denominator is 0.
- Constraints operators: All constraints operators extend BaseRelationConstraint: *OrConstraint*, *AndConstraint*, *LessConstraint*, *LessEqualConstraint*
- Tolerance-aware constraint operators: *EqualConstraint* allows a tolerance value in assessing the equality.. *==* can also be used for no tolerance equalities.


## Loading existing definitions

You can build your own constrained dataset by removing or adding constraints to an existing one.
In LCLD dataset, the term can only be 36 or 60 months, this is the constraint at index 3. 
We can replace it with a new set of constraints as follows (It is recommended to extend the whole class for a different set of constraints).

```python
from tabularbench.constraints.relation_constraint import Feature, Constant
from tabularbench.datasets.samples.lcld import get_relation_constraints

lcld_constraints =  get_relation_constraints()

new_constraint = (Feature("term") == Constant(36)) | (Feature("term") == Constant(48)) | (Feature("term") == Constant(60))
lcld_constraints[3] = new_constraint
```

## Constraint evaluation

Given a dataset, one can check the constraint satisfaction over all constraints, given a tolerance.

```python
from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory

tolerance=0.001
dataset = dataset_factory.get_dataset("url")
x, _ = dataset.get_x_y()

constraints_checker = ConstraintChecker(
    dataset.get_constraints(), tolerance
)
out = constraints_checker.check_constraints(x.to_numpy())
```

## Constraint repair

In the provided datasets, all constraints are satisfied. During the attack, Constraints can be fixed as follows:

```python
import numpy as np
from tabularbench.constraints.constraints_fixer import ConstraintsFixer
from tabularbench.constraints.relation_constraint import Feature

x = np.arange(9).reshape(3, 3)
constraint = Feature(0) == Feature(1) + Feature(2)

constraints_fixer = ConstraintsFixer(
            guard_constraints=[constraint],
            fix_constraints=[constraint],
        )

x_fixed = constraints_fixer.fix(x)

x_expected = np.array([[3, 1, 2], [9, 4, 5], [15, 7, 8]])

assert np.equal(x_fixed, x_expected).all()

```

Constraint violations can be translated into losses and you can compute the gradient to repair the faulty constraints as follows:

```python
import torch

from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)

from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.datasets.dataset_factory import get_dataset

ds = get_dataset("url")
constraints = ds.get_constraints()
constraint1 = constraints.relation_constraints[0]

x, y = ds.get_x_y()
x_metadata = ds.get_metadata(only_x=True)
x = torch.tensor(x.values, dtype=torch.float32)

constraints_executor = ConstraintsExecutor(
    constraint1,
    PytorchBackend(),
    feature_names=x_metadata["feature"].to_list(),
)
        
x.requires_grad = True
loss = constraints_executor.execute(x)
grad = torch.autograd.grad(
    loss.sum(),
    x,
)[0]

```