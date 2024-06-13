#!/bin/bash
docker run --rm -v "$(pwd):/workspace" tabularbench python -m tasks.train_model
