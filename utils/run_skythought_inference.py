"""
Lightweight runner to query SkyThought models (e.g., NovaSky-AI/Sky-T1-32B-Flash)
for reasoning/decision-style prompts using Hugging Face transformers.

Example:
  python utils/run_skythought_inference.py \\
      --model NovaSky-AI/Sky-T1-32B-Flash \\
      --prompt "Should we increase GBPUSD exposure this hour given rising volatility?"
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "NovaSky-AI/Sky-T1-32B-Flash"

# Default SkyThought system prompt taken from the published config.
SKYTHOUGHT_SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through a "
    "systematic long thinking process before providing the final precise and "
    "accurate solutions. This requires engaging in a comprehensive cycle of "
    "analysis, summarizing, exploration, reassessment, reflection, backtracing, "
    "and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified "
    "format:\n"
    "<|begin_of_thought|>\n"
    "{thought with steps separated with '\\n\\n'}\n"
    "<|end_of_thought|>\n"
    "Each step should include detailed considerations such as analysing questions, "
    "summarizing relevant findings, brainstorming new ideas, verifying the accuracy "
    "of the current steps, refining any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and "
    "reflections from the Thought section, systematically present the final "
    "solution that you deem correct. The solution should remain a logical, "
    "accurate, concise expression style and detail necessary step needed to reach "
    "the conclusion, formatted as follows:\n"
    "<|begin_of_solution|>\n"
    "{final formatted, precise, and clear solution}\n"
    "<|end_of_solution|>\n"
    "Now, try to solve the following question through the above guidelines:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkyThought model inference for a custom prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--prompt",
        help="Inline prompt text to send to the model. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing the prompt. Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the default SkyThought system prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to transformers (e.g., "auto", "cuda", "cpu").',
    )
    return parser.parse_args()


def load_prompt(prompt: Optional[str], prompt_file: Optional[Path]) -> str:
    if prompt and prompt_file:
        raise ValueError("Provide exactly one of --prompt or --prompt-file.")
    if prompt_file:
        return prompt_file.read_text()
    if prompt:
        return prompt
    raise ValueError("A prompt is required. Use --prompt or --prompt-file.")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    user_prompt = load_prompt(args.prompt, args.prompt_file)
    system_prompt = args.system_prompt or SKYTHOUGHT_SYSTEM_PROMPT
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nAssistant:"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and tokenizer.pad_token_id is not None:
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide an EOS or PAD token id for generation.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(completion.strip())


if __name__ == "__main__":
    main()
