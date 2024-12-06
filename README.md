# SemEval 2025 Task 10 - Narrative Extraction
## Multilingual Characterization and Extraction of Narratives from Online News

This repository contains the implementation for our submission to SemEval 2025 Task 10, Subtask C, focusing on narrative justification generation from news articles using Gemma 2 with instruction tuning and KTO alignment.

### Overview

Our approach achieves a BERTScore F1 of 0.74607 through a two-stage fine-tuning process:
1. Instruction tuning using QLoRA
2. KTO (Kahneman-Tversky Optimization) alignment with BERTScore as the reward function

### Installation

To install the `supertrainer` package:

```bash
git clone -b nlp-assignment-2 https://github.com/Erland366/supertrainer
cd supertrainer
make install_conda_env
```

Now the environment is installed on your hardware and you're ready to train the model.

### Training

#### 1. Instruction Tuning

To run the instruction tuning phase:

```bash
python src/supertrainer/train.py +experiments/semevaltask3=train_gemma2
```

#### 2. KTO Alignment

To run the KTO alignment phase:

```bash
python src/supertrainer/train.py +experiments/semevaltask3=train_kto_gemma2
```

### Results

Our implementation achieved the following performance metrics:

| Method | Precision | Recall | F1 Macro |
|--------|-----------|---------|-----------|
| Baseline (Phi) | 0.65540 | 0.67957 | 0.66719 |
| Instruct | 0.72135 | 0.69827 | 0.70921 |
| Fine-tune | 0.75833 | 0.73308 | 0.74515 |
| KTO | 0.75922 | 0.73416 | 0.74607 |

### Contact

Erland Hilman Fuadi  
MBZUAI  
erland.fuadi@mbzuai.ac.ae
