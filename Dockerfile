# The builder image, used to build the virtual environment
FROM python:3.8-buster as builder

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-103

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /environment

COPY pyproject.toml poetry.lock ./

RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.8-buster as runtime

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-103

WORKDIR /workspace

ENV VIRTUAL_ENV=/environment/.venv \
    PATH="/environment/.venv/bin:$PATH"

# RUN poetry install --without dev

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}


CMD ["python", "-m", "tasks.run_benchmark"]
