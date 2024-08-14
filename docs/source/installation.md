# Installation


## Using Docker (recommended)

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

## Using Pip

We recommend using Python 3.8.10.

1. Install the package from PyPI

    ```bash
    pip install tabularbench
    ```

## With Pyenv and Poetry

1. Clone the [repository](https://github.com/serval-uni-lu/tabularbench) 

2. Create a virtual environment using [Pyenv](https://github.com/pyenv/pyenv) with Python 3.8.10.

3. Install the dependencies using [Poetry](https://python-poetry.org/).

 ```bash
    poetry install
 ```

## Using conda

1. Clone the [repository](https://github.com/serval-uni-lu/tabularbench) 

2. Create a virtual environment using [Conda](https://docs.anaconda.com/free/miniconda/) with Python 3.8.10.

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
