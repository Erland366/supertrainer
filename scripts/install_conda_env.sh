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

# Default values
DEFAULT_ENV="supertrainer"
DEFAULT_FILE="environment.yaml"

# Parameters
CONDA_ENV=${1:-$DEFAULT_ENV}
ENV_FILE=${2:-$DEFAULT_FILE}

echo "Starting installation of conda environment: $CONDA_ENV from file: $ENV_FILE..."

# Install the conda environment from the specified YAML file
echo "Creating conda environment '$CONDA_ENV' from '$ENV_FILE'..."
conda env create --name "$CONDA_ENV" --file="$ENV_FILE"

# Activate the newly created environment
echo "Activating conda environment '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# Install the current project in editable mode
echo "Installing the current project in editable mode..."
pip install -e .

echo "Conda environment '$CONDA_ENV' installation and project setup completed."

# Deactivate the environment
conda deactivate

echo "Environment deactivated. To use this environment, run: conda activate $CONDA_ENV"