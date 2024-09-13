from contextlib import contextmanager
from typing import Any

from addict import Dict
from hydra import compose, initialize


def print_in_test(object):
    """Need to add new line for better formatting in test"""
    print()
    print(object)


@contextmanager
def hydra_config_context(
    overrides: list[str], config_path: str = "../configs", config_name: str = "train"
) -> Any:
    with initialize(config_path=config_path):
        cfg = Dict(dict(compose(config_name=config_name, overrides=overrides)))
        print_in_test(cfg)
        yield cfg
