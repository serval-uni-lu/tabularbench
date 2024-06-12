# TabularBench

TabularBench: Adversarial robustness benchmark for tabular data

**Leaderboard**: [https://serval-uni-lu.github.io/tabularbench/](https://serval-uni-lu.github.io/tabularbench/)

**Paper**: under review, preprint available shortly.

## Installation

### Using Docker (recommended)

1. Clone the repository

2. Build the Docker image

    ```bash
    ./tasks/docker_build.sh
    ```

3. Run the Docker container

    ```bash
    ./tasks/run_benchmark.sh
    ```

Note: The `./tasks/run_benchmark.sh` script mounts the current directory to the `/workspace` directory in the Docker container.
This allows you to edit the code on your host machine and run the code in the Docker container without rebuilding.

### With Pyenv and Poetry

1. Clone the repository

2. Create a virtual environment using [Pyenv](https://github.com/pyenv/pyenv) with python 3.8.10.

3. Install the dependencies using [Poetry](https://python-poetry.org/).

    ```bash
    poetry install
    ```

### Using conda (untested)

1. Clone the repository

2. Create a virtual environment using [Conda](https://docs.anaconda.com/free/miniconda/) with python 3.8.10.

    ```bash
    conda create -n tabularbench python=3.8.10
    ```

3. Activate the conda environment.

    ```bash
    conda activate tabularbench
    ```

4. Install the dependencies using Pip.

    ```bash
    pip install -r requirements.txt
    ```

## How to use

### Run the benchmark

You can run the benchmark with the following command:

```bash
python -m tasks.run_benchmark
```

or with Docker:

```bash
docker_run_benchmark
```

### Using the API

You can also use the API to run the benchmark. See `tasks/run_benchmark.py` for an example.

```python
clean_acc, robust_acc = benchmark(
    dataset="URL",
    model="STG_Default",
    distance="L2",
    constraints=True,
)
```

### Retrain the models

We provide the models and parameters used in the paper.
You can retrain the models with the following command:

```bash
python -m tasks.train_model
```

Edit the `tasks/train_model.py` file to change the model, dataset, and training method.

## Naming

For technical reasons the name of datasets, models, and training methods are different from the paper.
The mapping can be found in [doc/naming.md](doc/naming.md).
