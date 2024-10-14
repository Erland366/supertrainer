# MIT License
#
# Copyright (c) 2024 Edd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# !/bin/bash
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

OUTPUT_FILE="$1"


echo "Removing existing $OUTPUT_FILE if it exists..."
rm -f $OUTPUT_FILE


if [ -z "$1" ]; then
    echo "Error: No output file name provided."
    echo "Usage: $0 <output_file>"
    exit 1
fi


echo "Starting export of conda environment..."


echo "Removing existing $OUTPUT_FILE if it exists..."
rm -f "$OUTPUT_FILE"


if [ -f "setup.py" ]; then
    PROJECT_NAME=$(python -c "import setuptools; setup_cfg = setuptools.config.read_configuration('setup.py'); print(setup_cfg['metadata']['name'])")
elif [ -f "pyproject.toml" ]; then
    PROJECT_NAME=$(awk -F= '/^\[project\]/,/^name *=/ {if ($1 ~ /^ *name *$/) print $2}' pyproject.toml | tr -d ' "')
else
    echo "setup.py or pyproject.toml not found. Cannot determine project name."
    exit 1
fi

echo "Project name identified as: $PROJECT_NAME"


echo "Exporting conda environment to $OUTPUT_FILE..."
conda env export --no-builds | grep -v "^prefix: " | grep -v "^name: " | grep -v "$PROJECT_NAME" > "$OUTPUT_FILE"

echo "Conda environment export completed."
