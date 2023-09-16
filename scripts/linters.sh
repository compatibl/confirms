#!/bin/bash

cd ..

# Format using isort
echo "Format using isort"
isort confirms --sp=.isort.cfg
isort tests --sp=.isort.cfg

# Format using black
echo ""
echo "Format using black"
black -q confirms --config=pyproject.toml
black -q tests --config=pyproject.toml

# Validate using flake8
echo ""
echo "Validate using flake8"
flake8 confirms --config=.flake8
flake8 tests --config=.flake8

# Change back to the original directory
cd -
