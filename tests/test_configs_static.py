from __future__ import annotations

from pathlib import Path

import yaml


def test_cifar_configs_have_required_sections():
    for path in Path("configs/cifar_medium").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        for key in ("run", "data", "model", "diffusion", "training", "sampling"):
            assert key in config
        assert config["data"]["train_size"] == 45_000
        assert config["data"]["val_size"] == 5_000
        assert config["sampling"]["num_samples"] == 10_000
