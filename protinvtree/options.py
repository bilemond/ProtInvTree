"""CLI configuration shared by CATH evaluation and PDB inference."""

import argparse
import json
import math
from pathlib import Path

DEFAULTS = {
    "seed": 111, "total_steps": 4, "temperature": 0.3,
    "expand_n": 2, "pre_expand_n": 2, "rollout_n": 1,
    "simulations": 50, "pre_expansion_depth": 2, "c_param": 0.01,
    "value_threshold": 0.0, "complete_path_n": 1,
    "final_selection": "greedy",
    "selection": "random", "schedule": "cosine", "rollout_mode": "iterative",
    "reward_reference": "target", "reward_alpha": 1.0,
    "policy_model": "esm3-open", "fold_model": "facebook/esmfold_v1",
    "policy_device": "cuda", "reward_device": "cuda", "fold_chunk_size": 128,
}


def add_model_options(parser):
    parser.add_argument("--config", type=Path, help="JSON config; explicit CLI options override it")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)


def parse_options(parser, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    initial, _ = probe.parse_known_args(argv)
    if initial.config:
        with initial.config.open() as handle:
            defaults = json.load(handle)
        if not isinstance(defaults, dict):
            parser.error("Config must be a JSON object")
        unknown = set(defaults) - set(DEFAULTS)
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        for key, value in defaults.items():
            if type(value) is not type(DEFAULTS[key]):
                parser.error(f"Config {key} must be {type(DEFAULTS[key]).__name__}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    cfg = {key: getattr(args, key) for key in DEFAULTS}
    for key in ("total_steps", "expand_n", "pre_expand_n", "rollout_n", "complete_path_n"):
        if cfg[key] < 1:
            parser.error(f"{key} must be positive")
    if not 0 <= cfg["pre_expansion_depth"] <= cfg["total_steps"]:
        parser.error("pre-expansion-depth must lie between zero and total-steps")
    if cfg["simulations"] < 0 or cfg["fold_chunk_size"] < 0 or cfg["seed"] < 0:
        parser.error("simulations, fold-chunk-size and seed cannot be negative")
    for key in ("temperature", "c_param", "reward_alpha", "value_threshold"):
        if not math.isfinite(cfg[key]):
            parser.error(f"{key} must be finite")
    if cfg["temperature"] <= 0 or cfg["c_param"] < 0 or not 0 <= cfg["reward_alpha"] <= 1:
        parser.error("temperature must be positive; c-param >= 0; reward-alpha in [0, 1]")
    choices = {"selection": ("random", "entropy", "ar"), "schedule": ("cosine", "linear"),
               "rollout_mode": ("iterative", "jumpy"), "reward_reference": ("target", "folded-native"),
               "final_selection": ("greedy", "best-terminal")}
    for key, values in choices.items():
        if cfg[key] not in values:
            parser.error(f"{key} must be one of {values}")
    return args, cfg
