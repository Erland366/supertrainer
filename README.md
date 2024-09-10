# Preskripsi Training

This is a project for training Preskripsi models.

## Project Structure

The project has the following directory structure:

```
preskripsi-training
├── preskripsi_training
│   ├── __init__.py
│   ├── main.py
│   ├── config
│   │   ├── config.yaml
│   │   ├── sanity_check.yaml
│   │   └── __init__.py
│   ├── models
│   │   ├── model.py
│   │   └── __init__.py
│   ├── data
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── utils
│   │   ├── helpers.py
│   │   └── __init__.py
│   └── trainers
│       ├── base_trainer.py
│       ├── custom_trainer.py
│       └── __init__.py
├── tests
│   ├── test_model.py
│   ├── test_dataset.py
│   └── __init__.py
├── pyproject.toml
├── setup.py
└── README.md
```

## Usage

To run the project, execute the `main.py` script located in the `preskripsi_training` directory. Make sure to configure the project settings in the `config/config.yaml` file.

## Testing

The project includes unit tests for the model and dataset implementations. You can run the tests by executing the following commands:

```bash
pytest tests/test_model.py
pytest tests/test_dataset.py
```

## Installation

To install the project and its dependencies, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/preskripsi-training.git
```

2. Navigate to the project directory:

```bash
cd preskripsi-training
```

3. Install the project in editable mode:

```bash
pip install -e .
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```

Please note that you need to replace `your-username` in the installation instructions with your actual GitHub username or the appropriate repository URL.