#!/bin/bash

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

# Define the list of evaluations and dataset splits
evaluations=(
    "sonnet_instructor_factcheck"
    "gpt_instructor_factcheck"
    "qwen25_outlines_factcheck"
    "mistral03_outlines_factcheck"
    "llama32_outlines_factcheck"
    "gemma2_outlines_factcheck"
    "arabicbert_factcheck"
    "mbertuncased_factcheck"
    "indobertuncased_factcheck"
    "xlmr_factcheck"
)

dataset_splits=(
    "train_claim_en_evidence_en"
    "train_claim_en_evidence_idn"
    "train_claim_en_evidence_arb"
    "train_claim_idn_evidence_en"
    "train_claim_idn_evidence_idn"
    "train_claim_idn_evidence_arb"
    "train_claim_arb_evidence_en"
    "train_claim_arb_evidence_idn"
    "train_claim_arb_evidence_arb"
)

# Loop through each combination of evaluation and dataset split
for evaluation in "${evaluations[@]}"; do
    for split in "${dataset_splits[@]}"; do
        # Construct the command
        command="python src/supertrainer/evaluation.py +evaluation=$evaluation +dataset.dataset_kwargs.split=$split"

        # Echo the command for easy monitoring
        echo "Running: $command"

        # Execute the command
        $command
    done
done
