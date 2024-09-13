# Supertrainer
AI701 Final Assignment

# Resources
- For moondream training, my main resource of this is [this youtube channel](https://www.youtube.com/watch?v=5rH_VjKXuzg)

# Debug!
Read How to run first before reading this. This is just a debug section, but I know probably you'll be lazy to read this if I put it below the how to run section. BUT THIS IS IMPORTANT.

To see what's out config is before running the model, you can use this command
```bash
python src/supertrainer/main.py --cfg job
```

Make sure that these commands is correct before running the model. Hydra uses `OmegaConf` for the config. But I casted it into `addict.Dict` since `OmegaConf` can't transfer object file. But now it introduce another problem where if, let's say you want to access something like `config.a.b.c.d.e.f.g` which probably doesn't exists, it'll return you an empty dictionary (I plan to changed this into `None` instead) since `addict.Dict` is very dynamic and not enforce you to purposely have the key beforehand. Maybe it's better to have it as static or maybe some kind of instantiation first before adding new value to the key? That's a good idea tbh I'll try find this way.

[NOTE -> I supposed to already handle this below error by my commit of 0.0.2 by using the `StrictDict`. But I'll put this below error temporarily since I haven't tested fully the `StrictDict`]
Oh, because it keeps returning empty dictionary to the key that did not exists, you will find a lot of error something like `TypeError: Dict object is not subscriptable` or `TypeError: Dict object do not have an attribute X`. This is because the key that you're trying to access is not exists. So make sure that you're accessing the correct key. As long as you understand that these errors means that the key is not exists in `addict.Dict` part, you're good to go.

# Install Environment
To install the environment, you can do `make install_conda_env CONDA_ENV=<YOUR PREFERED NAME>` to create the conda environment. By default, `CONDA_ENV=supertrainer` if not specified. We assume that you use MBZUAI HPC cluster to run this code. If not, you need to change the `CONDA_PATH` in `scripts/install_conda_env.sh` to your own conda path.

The script will install the environment based on `environment.yaml` file. It'll also install the `supertrainer` package in editable mode. So you can modify the code and it'll automatically updated in the environment.

If you need to install new library or package, you can do it by either using `pip` or `conda`. Just to make sure to export your `environment.yaml` by doing `make export_conda_env` so that the environment is updated. This command will do an agnostic OS export + remove the `supertrainer` package from the `environment.yaml` file.

# How to run
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

TODO: CHECK BELOW COMMAND AGAIN THIS IS WEIRD
```bash
python src/supertrainer/main.py +experiment=train_mllm ++trainer.model_kwargs@trainer.model_kwargs=sdpa
```

It'll automatically use the `sdpa` file inside the `model_kwargs` folder.

You can also specify one single params like this
```bash
python src/supertrainer/main.py +experiment=train_mllm trainer.training_kwargs.num_train_epochs=10
```

Now our run will ran for 10 epochs.
