#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — Tego LoRA supervised fine-tuning

The training sample contains:
    - original_Wyck-SEQ
    - original_mag_density
    - candidate_mag_density
    - action

The prompt tokens are masked with labels = -100, while the action tokens
participate in the causal language-model teacher-forcing loss.

Example (single GPU):
    python train.py \
      --model_path /path/to/Llama-3.1-8B-Instruct \
      --train_csv /path/to/train.csv \
      --output_dir outputs/train \
      --num_train_epochs 10 \
      --precision bf16

For multi-GPU training, launch the same command with torchrun. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None


# ============================================================
# Constants
# ============================================================

DEFAULT_OUTPUT_DIR = "outputs/train"

LOGGER = logging.getLogger("tego.train")


# ============================================================
# Prompt template
# ============================================================

SYSTEM_PROMPT = r"""
You are a crystal-material inverse-design action generator.

Your task is to read an initial crystal structure, its magnetic density,
and a target magnetic density. Then you must output exactly one local structure edit action, enable the structure obtained after this editing operation attains the target magnetic density.

The initial structure input will be represented in the following Wyck-SEQ format:
1. [SPG=number@symbol] gives the space group number and Hermann-Mauguin symbol.
2. [LATTICE ...] gives the lattice parameters a, b, c, alpha, beta, gamma.
3. Each <WYCK_id=...> block describes one Wyckoff-equivalent atomic group.
4. In each Wyckoff block:
   - WYCK_id is the group index.
   - mult is the Wyckoff multiplicity.
   - wy is the Wyckoff letter.
   - sym is the site symmetry.
   - species is the chemical element occupying this Wyckoff group.
   - The following <idx=... x=... y=... z=...> lines are fractional coordinates
     of all equivalent sites in this Wyckoff group.
5. The structure begins with <BOS> and ends with <EOS>.

Action format:
You must output only one action block in the following format:

<ACTION>
<WYCK_id=... mult=... wy="..." sym="..." species="...">
<\ACTION>

- A valid action changes the species of exactly one whole Wyckoff block.
- The coordinates, lattice, multiplicity, Wyckoff letter, and site symmetry must not be changed by the action.

Strict output rules:
1. Output only one ACTION block. Do not output explanations.
2. Choose an existing WYCK_id from the input structure.
3. Prefer substitutions that are likely to preserve charge balance and structural stability.
4. For magnetic-density enhancement, transition metals and rare-earth elements may be useful, but they should still be selected under symmetry and stability constraints.
5. Avoid actions that yield a pure single-element crystal.
""".strip()


def build_user_prompt(
    original_wyck_seq: str,
    original_mag_density: str,
    candidate_mag_density: str,
) -> str:
    return f"""
Input:

<ORIGINAL_WYCK_SEQ>
{original_wyck_seq}
</ORIGINAL_WYCK_SEQ>

<ORIGINAL_MAG_DENSITY>
{original_mag_density}
</ORIGINAL_MAG_DENSITY>

<TARGET_MAG_DENSITY>
{candidate_mag_density}
</TARGET_MAG_DENSITY>

Now output the single best action.
""".strip()


# ============================================================
# Logging and distributed utilities
# ============================================================

def is_main_process() -> bool:
    try:
        import torch.distributed as dist

        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
    except Exception:
        return True


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    LOGGER.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if is_main_process():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        os.path.join(output_dir, "train.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


def safe_csv_field_size_limit():
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        try:
            csv.field_size_limit(2**31 - 1)
        except Exception:
            pass


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None

    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def format_float_or_unknown(x: Any, ndigits: int) -> str:
    v = parse_float(x)
    if v is None:
        return "unknown"
    return f"{v:.{ndigits}f}"


def validate_local_model_path(model_path: str):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            "\nLocal model_path does not exist:\n"
            f"{model_path}\n\n"
            "Please provide a valid local Hugging Face model directory containing config.json.\n"
        )

    required_candidates = [
        "config.json",
    ]

    missing = []
    for name in required_candidates:
        if not os.path.exists(os.path.join(model_path, name)):
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "\nmodel_path exists, but required model files are missing:\n"
            f"model_path = {model_path}\n"
            f"missing = {missing}\n"
        )


# ============================================================
# CSV loading
# ============================================================

@dataclass
class SFTSample:
    original_wyck_seq: str
    original_mag_density: str
    candidate_mag_density: str
    action: str


def normalize_action(action: Any) -> str:
    text = clean_str(action).strip()

    if not text:
        return ""

    if not text.startswith("<ACTION>"):
        text = "<ACTION>\n" + text

    if ("<\\ACTION>" not in text) and ("</ACTION>" not in text):
        text = text.rstrip() + "\n<\\ACTION>"

    return text.strip()


def load_sft_samples_from_csv(
    train_csv: str,
    original_col: str,
    original_mag_col: str,
    target_mag_col: str,
    action_col: str,
    max_rows: Optional[int] = None,
) -> List[SFTSample]:
    safe_csv_field_size_limit()

    samples: List[SFTSample] = []

    num_total = 0
    num_skip_missing = 0
    num_skip_empty = 0

    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(f"CSV header not found: {train_csv}")

        fieldnames = list(reader.fieldnames)
        required_cols = [original_col, original_mag_col, target_mag_col, action_col]
        missing_cols = [c for c in required_cols if c not in fieldnames]

        if missing_cols:
            raise ValueError(
                "CSV缺少必要列：\n"
                f"missing_cols = {missing_cols}\n"
                f"current_cols = {fieldnames}\n"
            )

        iterator = tqdm(
            reader,
            desc=f"Loading CSV: {os.path.basename(train_csv)}",
            disable=not is_main_process(),
        )

        for row in iterator:
            num_total += 1

            if max_rows is not None and len(samples) >= max_rows:
                break

            original_wyck_seq = clean_str(row.get(original_col, "")).strip()
            original_mag_density = format_float_or_unknown(row.get(original_mag_col), 4)
            candidate_mag_density = format_float_or_unknown(row.get(target_mag_col), 3)
            action = normalize_action(row.get(action_col, ""))

            if original_wyck_seq == "" or action == "":
                num_skip_empty += 1
                continue

            if original_mag_density == "unknown" or candidate_mag_density == "unknown":
                num_skip_missing += 1
                continue

            samples.append(
                SFTSample(
                    original_wyck_seq=original_wyck_seq,
                    original_mag_density=original_mag_density,
                    candidate_mag_density=candidate_mag_density,
                    action=action,
                )
            )

    if is_main_process():
        LOGGER.info(
            "CSV summary: total=%d, kept=%d, skip_empty=%d, skip_bad_mag_density=%d",
            num_total,
            len(samples),
            num_skip_empty,
            num_skip_missing,
        )

    if not samples:
        raise RuntimeError(
            "No usable samples after loading CSV. "
            "Please check column names and empty values."
        )

    return samples


# ============================================================
# Chat template and tokenization
# ============================================================

def build_prompt_text(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    bos = tokenizer.bos_token or ""
    return (
        f"{bos}System:\n{SYSTEM_PROMPT}\n\n"
        f"User:\n{user_prompt}\n\n"
        f"Assistant:\n"
    )


def build_full_text(tokenizer, user_prompt: str, action: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": action},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    eos = tokenizer.eos_token or ""
    return build_prompt_text(tokenizer, user_prompt) + action + eos


def tokenize_sft_example(
    tokenizer,
    sample: SFTSample,
    max_seq_length: int,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        original_wyck_seq=sample.original_wyck_seq,
        original_mag_density=sample.original_mag_density,
        candidate_mag_density=sample.candidate_mag_density,
    )

    prompt_text = build_prompt_text(tokenizer, user_prompt)
    full_text = build_full_text(tokenizer, user_prompt, sample.action)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    ).input_ids

    # 正常情况下 full_ids = prompt_ids + assistant_action_ids
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        target_ids = full_ids[len(prompt_ids):]
    else:
        # 兼容少数 tokenizer 的 chat_template 差异
        eos = tokenizer.eos_token or ""
        target_ids = tokenizer(
            sample.action + eos,
            add_special_tokens=False,
        ).input_ids

    if len(target_ids) == 0:
        target_ids = tokenizer(
            sample.action,
            add_special_tokens=False,
        ).input_ids

    if len(target_ids) == 0:
        raise RuntimeError("Empty target_ids. Please check action column.")

    # 超长处理：优先保留 action，prompt 太长则从前面截掉
    total_len = len(prompt_ids) + len(target_ids)

    was_truncated = False

    if total_len > max_seq_length:
        was_truncated = True

        keep_prompt_len = max_seq_length - len(target_ids)

        if keep_prompt_len > 0:
            prompt_ids = prompt_ids[-keep_prompt_len:]
        else:
            # 极端情况：target 本身都接近超过 max_seq_length
            # 保留 target 尾部，同时至少留 1 个 prompt token
            target_keep = max_seq_length - 1
            target_ids = target_ids[-target_keep:]
            prompt_ids = prompt_ids[-1:]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": len(input_ids),
        "target_len": len(target_ids),
        "was_truncated": was_truncated,
    }


class TegoActionDataset(Dataset):
    def __init__(
        self,
        samples: List[SFTSample],
        tokenizer,
        max_seq_length: int,
        log_first_n: int = 2,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.log_first_n = log_first_n

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = tokenize_sft_example(
            tokenizer=self.tokenizer,
            sample=self.samples[idx],
            max_seq_length=self.max_seq_length,
        )

        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }

    def inspect_lengths(self) -> Dict[str, Any]:
        lengths = []
        target_lengths = []
        truncated = 0

        for sample in tqdm(
            self.samples,
            desc="Inspecting token lengths",
            disable=not is_main_process(),
        ):
            item = tokenize_sft_example(
                tokenizer=self.tokenizer,
                sample=sample,
                max_seq_length=self.max_seq_length,
            )
            lengths.append(item["seq_len"])
            target_lengths.append(item["target_len"])
            truncated += int(item["was_truncated"])

        return {
            "num_samples": len(lengths),
            "seq_len_min": min(lengths),
            "seq_len_max": max(lengths),
            "seq_len_mean": sum(lengths) / len(lengths),
            "target_len_min": min(target_lengths),
            "target_len_max": max(target_lengths),
            "target_len_mean": sum(target_lengths) / len(target_lengths),
            "num_truncated": truncated,
        }


class SFTDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id

        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = int(math.ceil(max_len / m) * m)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            labels = f["labels"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ============================================================
# Callbacks
# ============================================================

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n: int):
        self.n = max(1, int(n))

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        ep = int(round(float(state.epoch)))

        if ep > 0 and ep % self.n == 0:
            control.should_save = True
        else:
            control.should_save = False

        return control

    def on_train_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "train_log.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        if logs is None:
            return control

        record = {
            "global_step": state.global_step,
            "epoch": state.epoch,
        }
        record.update(logs)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return control


# ============================================================
# Resume
# ============================================================

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None

    pattern = re.compile(r"^checkpoint-(\d+)$")

    best_step = -1
    best_path = None

    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is None:
            continue

        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue

        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path

    return best_path


# ============================================================
# Model loading
# ============================================================

def get_torch_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model(args):
    torch_dtype = get_torch_dtype(args.precision)

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "You used --use_4bit, but transformers.BitsAndBytesConfig "
                "cannot be imported. Please check transformers/bitsandbytes."
            ) from e

        compute_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

        model_kwargs["quantization_config"] = bnb_config

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model_kwargs["device_map"] = {"": local_rank}

        if is_main_process():
            LOGGER.info("Loading model in 4-bit QLoRA mode.")

    else:
        model_kwargs["torch_dtype"] = torch_dtype

        if is_main_process():
            LOGGER.info("Loading model in normal LoRA mode, dtype=%s.", str(torch_dtype))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    model.config.use_cache = False

    if args.use_4bit:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError(
                "prepare_model_for_kbit_training is unavailable. "
                "Please upgrade peft."
            )
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        except Exception as e:
            if is_main_process():
                LOGGER.warning("gradient_checkpointing_enable failed: %s", str(e))

        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    target_modules = [
        x.strip()
        for x in args.lora_target_modules.split(",")
        if x.strip()
    ]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_cfg)

    if is_main_process():
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # required paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_csv", "--dataset_csv", dest="train_csv", type=str, required=True)

    # output
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # csv columns
    parser.add_argument("--original_col", type=str, default="original_Wyck-SEQ")
    parser.add_argument("--original_mag_col", type=str, default="original_mag_density")
    parser.add_argument("--target_mag_col", type=str, default="candidate_mag_density")
    parser.add_argument("--action_col", type=str, default="action")

    # training aliases
    parser.add_argument("--num_train_epochs", "--epochs", dest="num_train_epochs", type=float, default=10)
    parser.add_argument("--max_seq_length", "--max_length", dest="max_seq_length", type=int, default=1300)
    parser.add_argument("--save_every_n_epochs", "--save_every_k_epochs", dest="save_every_n_epochs", type=int, default=1)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=500)

    # precision aliases
    parser.add_argument("--precision", "--dtype", dest="precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # memory
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # dataloader
    parser.add_argument("--num_workers", "--dataloader_num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None)

    # optimizer
    parser.add_argument("--optim", type=str, default=None)

    # resume
    parser.add_argument("--resume", "--resume_from_checkpoint", dest="resume", type=str, default="auto")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect_lengths", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    args.model_path = os.path.abspath(args.model_path)
    args.output_dir = os.path.abspath(args.output_dir)

    if args.optim is None:
        args.optim = "paged_adamw_8bit" if args.use_4bit else "adamw_torch"

    return args


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if is_main_process():
        LOGGER.info("=" * 100)
        LOGGER.info("Tego LoRA training")
        LOGGER.info("=" * 100)
        LOGGER.info("model_path repr = %r", args.model_path)
        LOGGER.info("train_csv  repr = %r", args.train_csv)
        LOGGER.info("output_dir repr = %r", args.output_dir)
        LOGGER.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    validate_local_model_path(args.model_path)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = load_tokenizer(args)

    # ----------------------------
    # Data
    # ----------------------------
    samples = load_sft_samples_from_csv(
        train_csv=args.train_csv,
        original_col=args.original_col,
        original_mag_col=args.original_mag_col,
        target_mag_col=args.target_mag_col,
        action_col=args.action_col,
        max_rows=args.max_rows,
    )

    train_dataset = TegoActionDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    if args.inspect_lengths:
        stats = train_dataset.inspect_lengths()
        if is_main_process():
            LOGGER.info("Token length stats:")
            LOGGER.info(json.dumps(stats, indent=2, ensure_ascii=False))

            stats_path = os.path.join(args.output_dir, "token_length_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            if stats["num_truncated"] > 0:
                LOGGER.warning(
                    "There are %d truncated samples. "
                    "Consider increasing --max_seq_length.",
                    stats["num_truncated"],
                )

    # ----------------------------
    # Model
    # ----------------------------
    model = load_model(args)

    # ----------------------------
    # Trainer
    # ----------------------------
    report_to = []
    if args.report_to and args.report_to.lower() != "none":
        report_to = [
            x.strip()
            for x in args.report_to.split(",")
            if x.strip()
        ]

    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"

    ddp_find_unused_parameters = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        save_strategy="epoch",
        save_total_limit=args.save_total_limit,

        fp16=use_fp16,
        bf16=use_bf16,
        tf32=args.tf32,

        optim=args.optim,

        report_to=report_to,
        remove_unused_columns=False,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,

        ddp_find_unused_parameters=ddp_find_unused_parameters if world_size > 1 else None,

        disable_tqdm=False,
    )

    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    def compute_metrics(eval_preds):
        # 这里只是让 loss 显示出来
        loss = eval_preds.loss if hasattr(eval_preds, "loss") else 0.0
        return {"train_loss": round(float(loss), 4)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[
            SaveEveryNEpochsCallback(args.save_every_n_epochs),
            JsonlLoggerCallback(args.output_dir),
        ],
        compute_metrics=compute_metrics,
    )

    # ----------------------------
    # Resume
    # ----------------------------
    resume_from = None

    if args.resume.lower() in {"off", "false", "none", "no"}:
        resume_from = None
    elif args.resume.lower() == "auto":
        resume_from = find_latest_checkpoint(args.output_dir)
    else:
        resume_from = args.resume

    if is_main_process():
        LOGGER.info("Resume mode: %s | resume_from=%s", args.resume, str(resume_from))

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ----------------------------
    # Final save
    # ----------------------------
    trainer.save_state()

    final_dir = os.path.join(args.output_dir, "final_lora")
    trainer.save_model(final_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

        metrics = train_result.metrics
        metrics_path = os.path.join(args.output_dir, "train_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        LOGGER.info("Final LoRA adapter saved to: %s", final_dir)
        LOGGER.info("Training metrics saved to: %s", metrics_path)
        LOGGER.info("[OK] Training finished.")


if __name__ == "__main__":
    main()