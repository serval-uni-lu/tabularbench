# Datasets

## Existing datasets

Our dataset factory support 5 datasets: CTU, LCLD, MALWARE, URL, and WIDS.
each dataset can be invoked with the following aliases:

```python
from tabularbench.datasets import dataset_factory

dataset_aliases= [
        "ctu_13_neris",
        "lcld_time",
        "malware",
        "url",
        "wids",
    ]

for dataset_name in dataset_aliases:
    dataset = dataset_factory.get_dataset(dataset_name)
    x, _ = dataset.get_x_y()
    metadata = dataset.get_metadata(only_x=True)
    assert x.shape[1] == metadata.shape[0]
```

## Building a new dataset

## Submitting a new dataset

Documention in progress.
