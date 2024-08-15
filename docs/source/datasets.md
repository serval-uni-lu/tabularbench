# Datasets

## Existing datasets

Our dataset factory support 5 datasets: CTU, LCLD, MALWARE, URL, and WIDS.
For each dataset, 

| Dataset | Based on                        | Task                      | Size      | # Features      | # Class balance | 
|---------|---------------------------------|---------------------------|-----------|-----------------|-----------------|
| LCLD    | George (2018)                   | Credit Scoring            | 1 220 092 | 28              | 80/20           |
| CTU-13  | Chernikova and Oprea (2022)     | Botnet Detection          | 198 128   | 756             | 99.3/0.7        |
| URL     | Hannousse and Yahiaouche (2021) | Phishing URL detection    | 11 430    | 63              | 50/50           |
| WIDS    | Lee et al. (2020)               | ICU patient survival      | 91 713    | 186        | 91.4/8.6        |
| MALWARE | Dyrmishi et al. (2022)          | Malware PE classification | 17 584    | 24 222             | 50/50           |

### Datasets descriptions
- **LCLD** (license: CC0: Public Domain) We develop a dataset derived from the publicly accessible Lending Club Loan Data. This dataset includes 151 features,
with each entry representing a loan approved by the Lending Club. However, some of these approved
loans are not repaid and are instead charged off. Our objective is to predict, at the time of the request,
whether the borrower will repay the loan or if it will be charged off. This dataset has been analyzed
by various practitioners on Kaggle. Nevertheless, the original dataset only contains raw data, and
to the best of our knowledge, there is no commonly used feature-engineered version. Specifically,
caution is needed when reusing feature-engineered versions, as many proposed versions exhibit data
leakage in the training set, making the prediction trivial. Therefore, we propose our own feature
engineering. The original dataset contains 151 features. We exclude examples where the feature
“loan status” is neither “Fully paid” nor “Charged Off,” as these are the only definitive statuses of
a loan; other values indicate an uncertain outcome. For our binary classifier, a “Fully paid” loan is
represented as 0, and a “Charged Off” loan is represented as 1. We begin by removing all features
that are missing in more than 30% of the examples in the training set. Additionally, we remove all
features that are not available at the time of the loan request to avoid bias. We impute features that
are redundant (e.g., grade and sub-grade) or too detailed (e.g., address) to be useful for classification.
Finally, we apply one-hot encoding to categorical features. We end up with 47 input features and one
target feature. We split the dataset using random sampling stratified by the target class, resulting in a
training set of 915K examples and a testing set of 305K examples. Both sets are unbalanced, with
only 20% of loans being charged off (class 1). We trained a neural network to classify accepted and
rejected loans, consisting of 3 fully connected hidden layers with 64, 32, and 16 neurons, respectively.
For each feature in this dataset, we define boundary constraints based on the extreme values observed
in the training set. We consider the 19 features under the control of the Lending Club as immutable.
We identify 10 relationship constraints (3 linear and 7 non-linear).
- **URL Phishing - ISCX-URL2016** (license CC BY 4.0) Phishing attacks are commonly employed
to perpetrate cyber fraud or identity theft. These attacks typically involve a URL that mimics a
legitimate one (e.g., a user’s preferred e-commerce site) but directs the user to a fraudulent website
that solicits personal or banking information. Hannousse and Yahiouche (2021) extracted features
from both legitimate and fraudulent URLs, as well as external service-based features, to develop a
classifier capable of distinguishing between fraudulent and legitimate URLs. The features extracted
from the URL include the number of special substrings such as “www”, “&”, “,”, “$”, “and”, the
length of the URL, the port, the presence of a brand in the domain, subdomain, or path, and the
inclusion of “http” or “https”. External service-based features include the Google index, page rank,
and the domain’s presence in DNS records. The full list of features is available in the reproduction
package. Hannousse and Yahiouche (2021) provide a dataset containing 5715 legitimate and 5715
malicious URLs. We use 75% of the dataset for training and validation, and the remaining 25% for
testing and adversarial generation. We extract a set of 14 relational constraints between the URL
features. Among these, 7 are linear constraints (e.g., the length of the hostname is less than or equal
to the length of the URL) and 7 are Boolean constraints of the form if a > 0 then b > 0 (e.g., if the
number of “http” > 0, then the number of slashes “/” > 0).
- **CTU-13** (license CC BY NC SA 4.0) This is a feature-engineered version of
CTU-13 proposed by Chernikova and Oprea (2019). It includes a combination of legitimate and
botnet traffic flows from the CTU University campus. Chernikova et al. aggregated raw network data
related to packets, duration, and bytes for each port from a list of commonly used ports. The dataset
consists of 143K training examples and 55K testing examples, with 0.74% of examples labeled as
botnet traffic (traffic generated by a botnet). The data contains 756 features, including 432 mutable
features. We identified two types of constraints that define what constitutes feasible traffic data. The
first type pertains to the number of connections and ensures that an attacker cannot reduce it. The
second type involves inherent constraints in network communications (e.g., the maximum packet size
for TCP/UDP ports is 1500 bytes). In total, we identified 360 constraints.
WiDS (license: PhysioNet Restricted Health Data License 1.5.0 1) Lee et al. (2020) dataset contains
medical data on the survival of patients admitted to the ICU. The objective is to predict whether
a patient will survive or die based on biological features (e.g., for triage). This highly unbalanced
dataset has 30 linear relational constraints.
- **Malware** (licence MIT) contains 24222 features extracted from a collection of benign and malware
Portable Executable (PE) files Dyrmishi et al. (2022). The objective of the classifier is to distinguish
between malware and benign software. The features include the DLL imports, the API
imports, PE sections, and statistic features such as the proportion of each possible byte value. The
dataset contains 17,584 samples. The number of total features (24 222) and the number of features involved in
each constraint (7 constraints) make this dataset challenging to attack. 

### Using an existing dataset

Each dataset can be invoked as follows:

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


### Datasets definitions

A dataset is defined by:

- an alias
- a data_source: generally based on a raw csv file, to be located by default in ./data/datasets/[alias]/[alias].csv
This csv file contains all the dataset samples
- a metadata_source: generally based on a meta-data csv file, to be located by default in ./data/datasets/[alias]/[alias]_metadata.csv
This csv file describe for each column of the raw csv file its
  - *type*: int (integer), real (float), cat (categorical), date (Date format)
  - *min*: lowest accepted boundary
  - *max* highest accepted boundary
  - *mutable* if this feature can be modified by the attacker
- a python definition file, to be located by default in ./tabularbench/datasets/samples/[alias].py

The definition file should contain at least:
```python
from typing import List
from tabularbench.datasets.dataset import (Dataset, CsvDataSource,
               DefaultIndexSorter, DefaultSplitter, Task)

from tabularbench.constraints.relation_constraint import (
    BaseRelationConstraint,
    Feature,
)

def get_relation_constraints() -> List[BaseRelationConstraint]:
    g1 = Feature(1) <= Feature(0)
    g2 = Feature(2) <= Feature(0)
    
    return [
        g1,
        g2,
    ]
def create_dataset() -> Dataset:
    
    tasks = [
        Task(
            name="my_task",
            task_type="classification",
            evaluation_metric="f1_score",
        )
    ]
    
    dataset = Dataset(
        name="[alias]",
        data_source=CsvDataSource(path="./data/datasets/[alias]/data.csv"),
        metadata_source=CsvDataSource(path="./data/datasets/[alias]/meta_data.csv"),
        tasks=tasks,
        sorter=DefaultIndexSorter(),
        splitter=DefaultSplitter(),
        relation_constraints=get_relation_constraints(),
    )

    return dataset

datasets = [
    {
        "name": "[alias]",
        "fun_create": create_dataset,
    },
]
```

where:
- data_source and metadata_source are classes that inherit DataSource, for example CsvDataSource class that loads a local csv file.
- tasks list the possible tasks for the given dataset (classification, regression, ...) and with the relevant evaluation metrics.
- sorter, a class to custom sort the samples of the dataset, the default sorter uses csv/dataframe index.
- splitter, a class that defines the split of the dataset; the default splitter splits into train(80%) and test (20%).
- relation_constraints, a method that defines the constraints of the dataset using the grammar defined in [tabularbench.constraints.relation_constraint](https://github.com/serval-uni-lu/tabularbench/blob/main/tabularbench/constraints/relation_constraint.py). Cf section "Constraints" for more details.

Examples of dataset definitions are located here for [URL](https://github.com/serval-uni-lu/tabularbench/blob/main/tabularbench/datasets/samples/url.py) and here for [LCLD](https://github.com/serval-uni-lu/tabularbench/blob/main/tabularbench/datasets/samples/url.py)

### Adding a dataset to the factory

1. Create a new definition file "weather.py" in the folder "./tabularbench/datasets/samples" using the alias "weather"
2. If your definition uses CsvDataSource, ensure your csv files are correctly located.
3. Add the "weather" datatset to the file "./tabularbench/datasets/samples/__init__.py" as follows:


```python
from tabularbench.datasets.samples.weather import datasets as weather_datasets

datasets = (
    ...
    + weather_datasets
)
```

4. You can now instantiate your new dataset with tht dataset factory:

```python
from tabularbench.datasets import dataset_factory

dataset = dataset_factory.get_dataset("weather")
x, y = dataset.get_x_y()
metadata = dataset.get_metadata(only_x=True)
```

## Submitting a new dataset

We welcome dataset contributions that bring additional insights or challenges to the tabular adversarial robustness community.

1. Create [a new issue](https://github.com/serval-uni-lu/tabularbench/issues/new/choose) by selecting the type "Submit a new dataset".
2. Fill in the form accordingly.
3. Create a new Pull Request with the dataset definition and files (csv, ...) and associate it with this issue.
4. We will validate that the dataset is working correctly, and run the architectures and defenses of the benchmark on it.
5. Once included, the dataset will be accessible both on the leaderboard and the dataset zoo.

If you find issues with existing datasets or if you want to suggest a variant of an existing dataset, you can raise an issue or use the same form to suggest an improved dataset.

Thank you for your contributions.
