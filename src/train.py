"""Model bootstrap entrypoint.

This script selects a model family from CLI, builds a default config, and
instantiates the corresponding model. Dataset/dataloader/training loop are
left for future implementation.
"""

import argparse
from dataclasses import asdict

from models.config import GPTConfig, LlamaConfig
from models.gpt import GPT
from models.llama import Llama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate a language model.")
    parser.add_argument(
        "model_name",
        choices=["gpt", "llama"],
        help="Model family to instantiate.",
    )
    return parser.parse_args()


def build_model_and_config(model_name: str) -> tuple[object, GPTConfig | LlamaConfig]:
    if model_name == "gpt":
        config = GPTConfig.default()
        return GPT(config), config
    if model_name == "llama":
        config = LlamaConfig.default()
        return Llama(config), config
    raise ValueError(f"Unsupported model_name: {model_name}")


def main() -> None:
    args = parse_args()
    model, config = build_model_and_config(args.model_name)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Selected model: {args.model_name}")
    print("Config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"Model parameter count: {num_params:,}")


if __name__ == "__main__":
    main()
