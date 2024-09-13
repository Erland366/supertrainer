#!/bin/bash
# Initialize conda
CONDA_PATH="/apps/local/anaconda2023/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "This script is configured to work only in the campus lab environment."
    echo "Please update the CONDA_PATH variable to point to your local conda installation."
    exit 1
fi

echo "Starting export of conda environment..."

# Remove the existing environment.yaml file if it exists
echo "Removing existing environment.yaml if it exists..."
rm -f environment.yaml

# Extract project name from setup.py or pyproject.toml
if [ -f "setup.py" ]; then
    PROJECT_NAME=$(python -c "import setuptools; setup_cfg = setuptools.config.read_configuration('setup.py'); print(setup_cfg['metadata']['name'])")
elif [ -f "pyproject.toml" ]; then
    PROJECT_NAME=$(awk -F= '/^\[project\]/,/^name *=/ {if ($1 ~ /^ *name *$/) print $2}' pyproject.toml | tr -d ' "')
else
    echo "setup.py or pyproject.toml not found. Cannot determine project name."
    exit 1
fi

echo "Project name identified as: $PROJECT_NAME"

# Export the conda environment to environment.yaml, excluding the prefix, name lines, and the project name
echo "Exporting conda environment to environment.yaml..."
conda env export --no-builds | grep -v "^prefix: " | grep -v "^name: " | grep -v "$PROJECT_NAME" > environment.yaml

echo "Conda environment export completed."