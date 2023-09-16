@echo off

pushd ..

echo Format using isort
isort confirms --sp=.isort.cfg
isort tests --sp=.isort.cfg

echo.
echo Format using black
black -q confirms --config=pyproject.toml
black -q tests --config=pyproject.toml

echo.
echo Validate using flake8
flake8 confirms --config=.flake8
flake8 tests --config=.flake8

popd
