import argparse
from pathlib import Path

import torch

from models.config import GPTConfig, LlamaConfig, ModelConfig
from models.gpt import GPT
from models.llama import Llama
from tokenizer import Tokenizer

CONTEXT_SIZE = 256


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Samples the next token from a logits tensor.

    Args:
        logits: Tensor of shape (batch_size, vocab_size).
                Note: Pass only the logits for the final token (e.g., logits[:, -1, :])
        temperature: Float > 0.0. Scales the logits. 0.0 equals greedy decoding.
        top_k: Integer. If > 0, keeps only the top K highest probability tokens.
        top_p: Float between 0 and 1. If < 1.0, applies nucleus sampling.

    Returns:
        next_token_ids: Tensor of shape (batch_size, 1) containing the sampled token IDs
    """
    # Fallback to greedy decoding in case temperature is zero.
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Scale logits by desired temperature.
    logits = logits / temperature
    probs = logits.softmax(dim=-1)

    # We perform top-k filtering before top-p to cheaply eliminate the massive tail of
    # the vocabulary.
    if top_k:
        # torch.topk already sorts the probs in descending order
        probs, indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
    elif top_p:
        # We need to sort probs in descending order for top-p operation
        probs, indices = torch.sort(probs, descending=True)

    if top_p:
        cum_probs = probs.cumsum(dim=-1)
        # If the cumulative probabilities of all tokens before the current token is
        # already larger than p, this token (and all tokens after it) should be excluded
        # from the candidate set.
        mask = (cum_probs - probs) > top_p
        probs[mask] = 0

    # torch.multinomial allows unnormalized weights in case top_p truncated them.
    next_token_ids = torch.multinomial(probs, num_samples=1)

    # Map back to original vocab indices if we altered the tensor shape/order
    if top_k or top_p:
        next_token_ids = torch.gather(indices, dim=-1, index=next_token_ids)

    return next_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive text generation CLI.")
    parser.add_argument(
        "model_name",
        choices=["gpt", "llama"],
        help=("Model family to load for generation."),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Model checkpoint path (default: checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("weights/bpe_tokenizer.json"),
        help="Tokenizer checkpoint path (default: weights/bpe_tokenizer.json).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff. Use 0 to disable top-k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold in (0,1]. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--max-generated-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per prompt.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")
    if args.max_generated_tokens <= 0:
        raise ValueError("--max-generated-tokens must be > 0")


def create_model_and_config(
    model_name: str, vocab_size: int
) -> tuple[torch.nn.Module, ModelConfig]:
    config: ModelConfig
    if model_name == "gpt":
        config = GPTConfig.default()
    elif model_name == "llama":
        config = LlamaConfig.default()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Per requirement: generation context is fixed to 256.
    config.max_seq_len = CONTEXT_SIZE
    config.vocab_size = max(config.vocab_size, vocab_size)
    config.use_cache = True

    if model_name == "gpt":
        return GPT(config), config
    return Llama(config), config


def select_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@torch.no_grad()
def generate_response(
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_generated_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    model.reset_cache()  # type: ignore

    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > CONTEXT_SIZE:
        raise ValueError(
            f"Prompt is too long: {len(input_ids)} tokens. "
            f"Maximum supported context is {CONTEXT_SIZE}."
        )

    eot_id = tokenizer.special_tokens.get("<|endoftext|>")
    top_k_arg = top_k if top_k > 0 else None
    top_p_arg = top_p if top_p < 1.0 else None

    # Inference state
    start_pos = 0
    generated_ids = []

    # Prefill input
    ctx_ids = input_ids[-CONTEXT_SIZE:]  # make sure the input fit in our context window
    x = torch.tensor([ctx_ids], dtype=torch.long, device=device)

    for _ in range(max_generated_tokens):
        logits = model(x, start_pos)
        start_pos += x.shape[1]
        # Only sample from the last token's logits
        next_token = sample_next_token(
            logits[:, -1, :], temperature, top_k_arg, top_p_arg
        )
        next_token_id = int(next_token.item())
        generated_ids.append(next_token_id)
        if eot_id is not None and next_token_id == eot_id:
            break

        x = next_token

    return tokenizer.decode(generated_ids)


def main() -> None:
    args = parse_args()
    validate_args(args)

    tokenizer = Tokenizer.load(args.tokenizer)
    vocab_size = max(tokenizer.unified_vocab.keys()) + 1
    model, config = create_model_and_config(args.model_name, vocab_size=vocab_size)

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device = select_device()
    model.to(device)
    model.eval()

    print(f"Loaded {args.model_name} on {device}.")
    print(f"Context size: {config.max_seq_len}")
    print("Type a prompt and press Enter. Type 'exit' to quit.")

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not prompt:
            continue

        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_generated_tokens=args.max_generated_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            continue

        print(f"Model: {response}")


if __name__ == "__main__":
    main()
