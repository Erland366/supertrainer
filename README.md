# Object Material Classification for Robotic Manipulation using Vision Language Models

<img src="./assets/imgs/AI701.drawio (1).svg">

This repository contains the implementation of our research on using Vision-Language Models (VLMs) for object material classification to enhance robotic manipulation. The project explores how different VLM architectures can be efficiently fine-tuned to classify object materials and integrate with robotic systems for adaptive gripping.

## Overview

Robotic manipulation of objects with diverse material properties (soft, hard, medium) presents significant challenges, particularly in applying uniform gripping force which can lead to object damage. Our research addresses this by:

1. Evaluating various VLM architectures for material classification
2. Implementing efficient fine-tuning using QLoRA
3. Integrating classification outputs with robotic control systems
4. Comparing performance against traditional CNN-based approaches

## Key Features

- Implementation of multiple VLM architectures:
  - Phi-3.5 Vision
  - DeepSeekVL
  - InternVL2

- Early Fusion models:
  - Deepseek Janus
  - Chameleon

- Encoder-Decoder model:
   - Instruct-BLIP

- Baseline CNN models:
  - ResNet-101
  - VGG16
  - EfficientNet-B0

- QLoRA (Quantized Low-Rank Adaptation) implementation for efficient fine-tuning
- Arduino-based robotic system integration
- Comprehensive evaluation framework

## Results

Our experiments show that fine-tuned VLMs significantly outperform traditional CNN models:

| Model Type | Model Name | Accuracy |
|------------|------------|----------|
| Heuristic | Most Common | 25.7% |
| CNN-based | ResNet-101 | 54.3% |
| CNN-based | VGG16 | 49.3% |
| CNN-based | EfficientNet-B0 | 57.4% |
| Encoder-Decoder | InstructBLIP | 74.6% |
| VLM | DeepseekVL | 76.2% |
| VLM | InternVL2 | 72.5% |
| VLM | Phi-3.5 Vision | 66.1% |
| Early Fusion | Chameleon | 43.7% |
| Early Fusion | Deepseek Janus | 73.5% |

The weight for DeepseekVL can be downloaded [here](https://drive.google.com/file/d/1V4rr-JovncbbHZFZAnFOotkPO1hZPQ76/view?usp=sharing)

## Dataset

We use a subset of the PhysObjects dataset, focusing on material classification:
- 6.83K training samples
- 1.12K test samples
- Available at: https://huggingface.co/datasets/Erland/AI701_project

## Requirements

- PyTorch
- Hugging Face Transformers
- Ms-Swift
- Arduino IDE
- NVIDIA GPU with 24GB+ memory (tested on Quadro RTX 6000 and RTX 4090)

## Hardware Setup

- Arduino UNO
- Servo tower-pro MG995 motors
- Serial communication interface

## Usage

### Installation

To install this `supertrainer` package. You can do it by :

```
git clone -b ai701 https://github.com/Erland366/supertrainer
cd supertrainer
make export_conda_env
```

Now environment is installed to your hardware and you ready to train the model

### Train

#### Chameleon

To train `chameleon` model, you can do it by :

```
python src/supertrainer/train.py +experiments/soft_robotics=train_chameleon
```

#### Phi-3.5 Vision

To train `Phi-3.5 Vision` model, you can do it by :

```
python src/supertrainer/train.py +experiments/soft_robotics=train_phi35vision
```

#### InstructBLIP

To train `InstructBLIP` model, you can do it by :

```
python src/supertrainer/train.py +experiments/soft_robotics=train_instruct_blip
```

#### DeepseekVL, InternVL, Deepseek Janus

To Train these models, first you need to adjust the training script in `scripts/ai701/train_swift.sh` and prepare the dataset first by following the guideline in [ms-swift documentation](https://swift.readthedocs.io/en/latest/). After that, you can run the training script by running this command :

```
./scripts/ai701/train_swift.sh <model_name>
```

### Evaluation

#### Chameleon

To evaluate `chameleon` model, you can do it by :

```
python src/supertrainer/evaluation.py +experiments/soft_robotics=evaluation_chameleon
```

To change the dataset, you can specify your dataset in `configs/dataset/mllm/soft_robotics.yaml` under `dataset_kwargs/path` key

To change the model adapter, you can specify your adapter in `configs/experiments/soft_robotics/evaluation_chameleon.yaml` under `evaluation/model_name` key.

#### Phi-3.5 Vision

To evaluate `Phi-3.5 Vision` model, you can do it by :

```
python src/supertrainer/evaluation.py +experiments/soft_robotics=evaluation_phi35vision
```

To change the dataset, you can specify your dataset in `configs/dataset/mllm/soft_robotics.yaml` under `dataset_kwargs/path` key

To change the model adapter, you can specify your adapter in `configs/experiments/soft_robotics/evaluation_phi35vision/model_name` key.

#### InstructBLIP

To evaluate `InstructBLIP` model, you can do it by :

```
python src/supertrainer/evaluation.py +experiments/soft_robotics=evaluation_instruct_blip
```

To change the dataset, you can specify your dataset in `configs/dataset/mllm/soft_robotics.yaml` under `dataset_kwargs/path` key

To change the model adapter, you can specify your adapter in `configs/experiments/soft_robotics/evaluation_instruct_blip.yaml` under `evaluation/model_name` key.

#### DeepseekVL, InternVL, Deepseek Janus
You can follow an example notebook in `notebooks/swift_inference.ipynb` and change the model accordingly.

<!-- ## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citations

If you use this code in your research, please cite our work:

[Note: Add citation details once published] -->

## Related Work

This project builds upon and extends the work from:
- Gao et al. (2023) - Physical Grounding of Vision-Language Models for Robotic Manipulation
- Various VLM architectures including Phi-3, DeepSeekVL, and InternVL

<!-- ## License

[Add appropriate license] -->

## Authors

- Erland Fuadi
- Nakul Nibe
- Rifo Genadi

## Acknowledgments

Mohamed Bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE

## Contact

For questions or feedback, please contact:
- erland.fuadi@mbzuai.ac.ae
- nakul.nibe@mbzuai.ac.ae
- rifo.genadi@mbzuai.ac.ae
