# Supertrainer

## Overview

Supertrainer is a unified repository for various trainers, inspired by projects like [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [Torchtune](https://github.com/pytorch/torchtune), and [Unsloth](https://github.com/unslothai/unsloth). It aims to provide a flexible and extensible framework for training, inferencing, and benchmarking different types of models (e.g., BERT, LLM, MLLM) using a configuration-driven approach.

## Resources
- For moondream training, my main resource of this is [this youtube channel](https://www.youtube.com/watch?v=5rH_VjKXuzg)

## Debug!
Read How to run first before reading this. This is just a debug section, but I know probably you'll be lazy to read this if I put it below the how to run section. BUT THIS IS IMPORTANT.

To see what's out config is before running the model, you can use this command
```bash
python src/supertrainer/main.py --cfg job
```

Make sure that these commands is correct before running the model. Hydra uses `OmegaConf` for the config. But I casted it into `addict.Dict` since `OmegaConf` can't transfer object file. But now it introduce another problem where if, let's say you want to access something like `config.a.b.c.d.e.f.g` which probably doesn't exists, it'll return you an empty dictionary (I plan to changed this into `None` instead) since `addict.Dict` is very dynamic and not enforce you to purposely have the key beforehand. Maybe it's better to have it as static or maybe some kind of instantiation first before adding new value to the key? That's a good idea tbh I'll try find this way.

[NOTE -> I supposed to already handle this below error by my commit of 0.0.2 by using the `StrictDict`. But I'll put this below error temporarily since I haven't tested fully the `StrictDict`]

Oh, because it keeps returning empty dictionary to the key that did not exists, you will find a lot of error something like `TypeError: Dict object is not subscriptable` or `TypeError: Dict object do not have an attribute X`. This is because the key that you're trying to access is not exists. So make sure that you're accessing the correct key. As long as you understand that these errors means that the key is not exists in `addict.Dict` part, you're good to go.

## Install Environment
To install the environment, you can do `make install_conda_env CONDA_ENV=<YOUR PREFERED NAME>` to create the conda environment. By default, `CONDA_ENV=supertrainer` if not specified. We assume that you use MBZUAI HPC cluster to run this code. If not, you need to change the `CONDA_PATH` in `scripts/install_conda_env.sh` to your own conda path.

The script will install the environment based on `environment.yaml` file. It'll also install the `supertrainer` package in editable mode. So you can modify the code and it'll automatically updated in the environment.

If you need to install new library or package, you can do it by either using `pip` or `conda`. Just to make sure to export your `environment.yaml` by doing `make export_conda_env` so that the environment is updated. This command will do an agnostic OS export + remove the `supertrainer` package from the `environment.yaml` file.

## How to run
Everything that you need for training is in `src/supertrainer/main.py`. So you can run this file to train the model.

```bash
python src/supertrainer/main.py
```

Because it's based on [Hydra](https://hydra.cc/), you can follow it's example to change any of the parameters. For example, since we will do a lot of training on different cases like maybe training BERT or training MLLM, you will be changing alot the experiment part of the config.

```bash
python src/supertrainer/main.py +experiment=training_bert
```

In Hydra, every configs are in `configs/` folder. So you can change the parameters there. My plan is to split the task to each of their own folder. For example I already implemented `bert/` and `mllm/` in here. Next milestone on this project will be :
 - [ ] Implementing LLM training

For each project, mainly they will have two folder, which is `dataset/` and `trainer/`. The `dataset/` will contain the dataset and the `trainer/` will contain the training loop. Maybe I'll add `model/` in here too if we need really custom model or want to tweak the architecture. Generally you don't but I am not too sure.

> What's the different between `trainer/` and `model/`? Well you can think of `trainer/` as in `Trainer` in HuggingFace where we only setup like `batch_size` or `learning_rate` stuff. We don't really modify any architecture inside it. Whereas `model/` is where maybe we want to change the activation function or the number of dimension of the model.

For `dataset/`, generally I found it simple where mostly I only need to specify the dataset path or the tokenizer if needed. But I open on suggestion on better design of this.

For `trainer/`, I found it a bit more complex. So since everything in here will be based on QLoRA, there are `bitsandbytes_kwargs/` which specify what kind of LoRA do we want to use, `peft_kwargs/` which tells which layer and ranks of the LoRA, `model_kwargs/` which specify the settings of the actual model, and `trainer_kwargs/` which specify the training loop.
- `bitsandbytes_kwargs/` -> I found this rare to modify, maybe we can add like LoRA 8-bits but generally it's not recommended. Since probably all of the settings will use the same config, I am thinking of making like `default/` folder and just use this settings from there.
- `model_kwargs/` -> I found that I never change this much. So I am also thinking to move this into like global config or someting.
- `peft_kwargs/` -> This is where we specify the layer and ranks of the LoRA. I found this is the most important part of the training since different model will use different sets of this. I highly suggested to naming each file of this to the model name. But me myself didn't apply that yet LOL
- `training_kwargs/` -> This is where we specify the training loop. I also didn't really change this much for each architecture. But 100% we will change a lot of this per-architecture. We can utilize Hydra capability for this.

For each of task category, I put a config file that combine the above config into one file. For example in `mllm/` I put `mllm_kwargs.yaml`. But I found that I keep putting custom config in here which probably is not modular at all? So maybe I should be separating per-architecture but then it'll overlap with `peft_kwargs/`?? Maybe we can set like default settings for the `peft_kwargs/` and only modify certain part per architecture? (we can do this in Hydra)

Let's say I want to modify something in the config. Let's say I want to use `sdpa` as my attention instead of `flash_attention_2` which is the default config. What can I did is to modify the training cli into like this.

```bash
python src/supertrainer/train.py +experiment=train_mllm trainer/common/model_kwargs@trainer.model_kwargs=sdpa
```

It'll automatically use the `sdpa` file inside the `model_kwargs` folder.

Note that the `@` symbol is used since we want to map the the file of `sdpa` from `trainer/common/model_kwargs` to `trainer.model_kwargs` (Note the usage of `/` and `.`. I know it's weird but I can't find other solution of this).

You can also specify one single params like this
```bash
python src/supertrainer/main.py +experiment=train_mllm trainer.training_kwargs.num_train_epochs=10
```

Now our run will ran for 10 epochs.


## Project Structure
<details>
<summary> Expand This to See The Structure Plan </summary>

```
supertrainer/
├── configs/
│   ├── experiment/
│   │   ├── train_mllm.yaml
│   │   ├── train_bert.yaml
│   │   └── train_llm.yaml
│   ├── trainer/
│   │   ├── common/
│   │   │   ├── hf_trainer_kwargs.yaml
│   │   │   └── bitsandbytes_kwargs.yaml
│   │   ├── mllm/
│   │   │   ├── model_kwargs.yaml
│   │   │   ├── peft_kwargs.yaml
│   │   │   └── training_kwargs.yaml
│   │   ├── bert/
│   │   │   ├── model_kwargs.yaml
│   │   │   ├── peft_kwargs.yaml
│   │   │   └── training_kwargs.yaml
│   │   └── llm/
│   │       ├── model_kwargs.yaml
│   │       ├── peft_kwargs.yaml
│   │       └── training_kwargs.yaml
│   ├── dataset/
│   │   ├── mllm/
│   │   ├── bert/
│   │   └── llm/
│   ├── model/
│   │   ├── mllm/
│   │   ├── bert/
│   │   └── llm/
│   └── mode/
│       ├── sanity_check.yaml
│       ├── toy.yaml
│       └── full.yaml
├── src/
│   ├── data/
│   │   ├── base.py
│   │   ├── mllm.py
│   │   ├── bert.py
│   │   ├── llm.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── base.py
│   │   ├── mllm.py
│   │   ├── bert.py
│   │   ├── llm.py
│   │   └── __init__.py
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   ├── mllm_trainer.py
│   │   ├── bert_trainer.py
│   │   ├── llm_trainer.py
│   │   └── __init__.py
│   ├── inference/
│   │   ├── base_inferencer.py
│   │   ├── mllm_inferencer.py
│   │   ├── bert_inferencer.py
│   │   ├── llm_inferencer.py
│   │   └── engines/
│   │       ├── hf_vanilla.py
│   │       ├── vllm.py
│   │       └── tensorrt.py
│   ├── benchmark/
│   │   ├── base_benchmark.py
│   │   ├── mllm_benchmark.py
│   │   ├── bert_benchmark.py
│   │   └── llm_benchmark.py
│   ├── metrics/
│   │   ├── base_metric.py
│   │   ├── classification_metrics.py
│   │   ├── generation_metrics.py
│   │   └── custom_metrics.py
│   ├── utils/
│   │   └── ...
│   ├── train.py
│   ├── inference.py
│   └── eval.py
└── ...
```
</details>

## Design Decisions and Rationale

1. **Configuration-Driven Approach**
   - Why: Allows for easy experimentation and reproducibility without changing code.
   - How: Using YAML files in the `configs/` directory, leveraging Hydra for composition.

2. **Separation of Concerns**
   - Why: Improves maintainability and allows for easy extension of functionality.
   - How: Separate directories for different components (data, models, trainers, etc.).

3. **Task-Specific Implementations**
   - Why: Different model types (BERT, LLM, MLLM) have unique requirements.
   - How: Task-specific classes (e.g., `BERTInferencer`, `LLMTrainer`) inherit from base classes.

4. **Flexible Inference Engines**
   - Why: Different inference engines (e.g., HuggingFace, vLLM, TensorRT) have varying performance characteristics.
   - How: Engine-specific configurations and implementations in `src/inference/engines/`.

5. **Unified Benchmarking**
   - Why: Consistent evaluation across different model types and inference engines.
   - How: Common benchmarking interface with task-specific implementations.

6. **Extensible Metrics**
   - Why: Different tasks require different evaluation metrics.
   - How: Modular metric implementations in `src/metrics/`.

7. **Single Entry Point**
    Why: Provides clear separation between different operations and allows for operation-specific configurations and logic.
    How: Separate scripts (`train.py`, `inference.py`, `benchmark.py`) in the scripts/ directory for each main operation.

8. **Model Source Flexibility**
   - Why: Support for both pre-trained models from HuggingFace and locally saved models.
   - How: Configuration options for model source and path.

9. **Customizable Training Loops**
   - Why: Different tasks and models may require specific training procedures.
   - How: Task-specific trainer implementations with shared base functionality.

## Key Features

- **Unified Interface**: Train, infer, and benchmark different model types using a consistent interface.
- **Configurability**: Easily switch between models, datasets, and training/inference settings using YAML configs.
- **Extensibility**: Add new model types, inference engines, or metrics with minimal changes to existing code.
- **Reproducibility**: Experiments can be easily reproduced by sharing configuration files.
- **Performance Optimization**: Support for various inference engines allows for optimized deployment.

## Next Milestone
- [ ] Implement LLM Training with Unsloth
- [ ] Refactor dataset to follow the structure plan
- [ ] Implement LLM Training with HuggingFace Trainer?

## Known Bugs
- [ ] I implemented `self.config.testing` but I don't really use it for the testing. Maybe I shoud really think better of this!
- [ ] In `mllm` we actually need to pass the column name `image_col` for `DataCollatorWithPadding`, sadly we only pass the config name to `dataset/` and not in `trainers/`. This means I have to explicitly pass the `image_col` in the `postprocess_config` of `mllm`. One solution that I can think of is to not separately the config between `trainer` and `dataset` (Even though currently it's not splitted in `.yaml`, it'll get splitted in `src/supertrainer/train.py`)
- [ ] Every single `push_to_hub` is still buggy :(. What I want is to be kinda like the [unsloth](https://github.com/unslothai/unsloth/blob/62c989ef0ae0e9fbac714a4cb21eda76c1fe84b6/unsloth/save.py#L183-L210) codebase where you can push the model to the hub while solving every single problem (basically just ready to inference).
   - One single "simple" example (only simple if you know the thing, if not you are screwed!) is to do padding to the left like [this](https://github.com/unslothai/unsloth/blob/62c989ef0ae0e9fbac714a4cb21eda76c1fe84b6/unsloth/save.py#L328-L329C37). Their repo is so gooodddd!!!!
- [ ] I need to consider more in `supertrainer/data/base.py` where for now I commented the `prepare_dataset` function since it's clash with `supertrainer/data/llm.py` of `ConversationLLMDataset`. I need better design for this!
- [ ] We haven't use the `tokenizer` instruct for the `llm` which is bad since we want that `chat_template`!
- [ ] When `sanity_check`, we should remove evaluaton dataset (maybe?). Which means we need to remove certain config like `eval_dataset`, `auto_batch_size`, etc
- [ ] Fix `FA2` in the `environment.yaml`

## Future Considerations

 - [ ] Fused Kernels
 - [ ] Early Fusion training support
 - [ ] Distributed training support
 - [ ] Automated hyperparameter tuning
 - [ ] Support for additional model architectures and tasks

By following this structure and philosophy, SuperTrainer aims to provide a flexible and powerful tool for researchers and practitioners working with a variety of machine learning models and tasks.
