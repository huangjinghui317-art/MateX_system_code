#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer.py — Tego iterative crystal inverse-design inference

Core workflow:
    CSV[cif]
    -> pymatgen Structure
    -> MatterSim initial energy per atom
    -> Wyck-SEQ encoding
    -> language model / LoRA generates a local edit action
    -> apply the action to one Wyckoff group
    -> MatterSim relaxation and energy filtering
    -> CHGNet magnetic-density prediction
    -> select the best retained candidate as the next-round input
    -> repeat for at most N rounds
    -> save CIFs, trajectories, CSV results, statistics, and stage timings

Example (single GPU):
    python infer.py \
      --model_path /path/to/Llama-3.1-8B-Instruct \
      --lora_path outputs/train/final_lora \
      --input_csv /path/to/input.csv \
      --cif_col cif \
      --output_dir outputs/infer \
      --target_mag_density 0.2 \
      --num_rounds 5 \
      --k 8

For multi-GPU inference, launch the same command with torchrun. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


# ============================================================
# Prompt: must be consistent with training
# ============================================================

SYSTEM_PROMPT = r"""You are a crystal-material inverse-design action generator.

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
5. Do NOT use radioactive elements including Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U.
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


# ============================================================
# Logging / distributed
# ============================================================

LOGGER = logging.getLogger("tego.infer")


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_if_needed():
    world_size = get_world_size()
    if world_size <= 1:
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def barrier_if_needed():
    if get_world_size() <= 1:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    rank = get_rank()

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s")

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    log_path = os.path.join(output_dir, f"infer_rank{rank}.log")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(RankFilter())
    LOGGER.addHandler(fh)

    if is_main_process():
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.addFilter(RankFilter())
        LOGGER.addHandler(sh)


# ============================================================
# Wyckoff conversion
# ============================================================

@dataclass
class WyckoffBlock:
    wyck_id: int
    mult: int
    wy: str
    sym: str
    species: str
    indices: list


def _symmetry_dataset_get(ds, key: str):
    try:
        return ds[key]
    except Exception:
        return getattr(ds, key)


def extract_wyckoff_blocks(
    structure,
    symprec=1e-2,
    angle_tolerance=5.0,
):
    sga = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    ds = sga.get_symmetry_dataset()

    wyckoffs = list(_symmetry_dataset_get(ds, "wyckoffs"))
    equiv_atoms = list(_symmetry_dataset_get(ds, "equivalent_atoms"))
    site_syms = list(_symmetry_dataset_get(ds, "site_symmetry_symbols"))

    groups = defaultdict(list)

    for i, site in enumerate(structure):
        key = (
            site.specie.symbol,
            wyckoffs[i],
            site_syms[i],
            int(equiv_atoms[i]),
        )
        groups[key].append(i)

    blocks = []

    for bid, ((species, wy, sym, eq), indices) in enumerate(groups.items()):
        blocks.append(
            WyckoffBlock(
                wyck_id=bid,
                mult=len(indices),
                wy=wy,
                sym=sym,
                species=species,
                indices=sorted(int(x) for x in indices),
            )
        )

    blocks.sort(key=lambda b: (b.species, b.wy, b.sym, b.indices[0]))

    for i, b in enumerate(blocks):
        b.wyck_id = i

    return blocks


def fmt_float(x, ndigits=3):
    return f"{float(x):.{ndigits}f}"


def get_spacegroup_text(structure, symprec=1e-2, angle_tolerance=5.0):
    try:
        sga = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        spg_num = sga.get_space_group_number()
        spg_symbol = sga.get_space_group_symbol()
        return spg_num, spg_symbol
    except Exception:
        return None, "UNKNOWN"


def structure_to_wyckoff_seq(
    structure,
    symprec=1e-2,
    angle_tolerance=5.0,
    ndigits=3,
):
    spg_num, spg_symbol = get_spacegroup_text(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    lattice = structure.lattice
    blocks = extract_wyckoff_blocks(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    lines = []

    lines.append("<BOS>")

    if spg_num is None:
        lines.append(f"[SPG=UNKNOWN@{spg_symbol}]")
    else:
        lines.append(f"[SPG={spg_num}@{spg_symbol}]")

    lines.append(
        "[LATTICE "
        f"a={fmt_float(lattice.a, ndigits)}, "
        f"b={fmt_float(lattice.b, ndigits)}, "
        f"c={fmt_float(lattice.c, ndigits)}, "
        f"alpha={fmt_float(lattice.alpha, ndigits)}, "
        f"beta={fmt_float(lattice.beta, ndigits)}, "
        f"gamma={fmt_float(lattice.gamma, ndigits)}]"
    )

    for block in blocks:
        lines.append(
            f'<WYCK_id={block.wyck_id} '
            f'mult={block.mult} '
            f'wy="{block.wy}" '
            f'sym="{block.sym}" '
            f'species="{block.species}">'
        )

        for idx in block.indices:
            site = structure[int(idx)]
            x, y, z = site.frac_coords
            x = x % 1.0
            y = y % 1.0
            z = z % 1.0

            lines.append(
                f'  <idx={int(idx)} '
                f'x={fmt_float(x, ndigits)} '
                f'y={fmt_float(y, ndigits)} '
                f'z={fmt_float(z, ndigits)}>'
            )

    lines.append("<EOS>")

    return "\n".join(lines)


def parse_structure_from_cif(cif_text):
    return Structure.from_str(str(cif_text), fmt="cif")


# ============================================================
# Action parsing / application
# ============================================================

def parse_action(action_text):
    pattern = (
        r"<WYCK_id=(?P<wyck_id>\d+)\s+"
        r"mult=(?P<mult>\d+)\s+"
        r'wy="(?P<wy>[^"]+)"\s+'
        r'sym="(?P<sym>[^"]+)"\s+'
        r'species="(?P<species>[^"]+)">'
    )

    m = re.search(pattern, str(action_text))

    if m is None:
        raise ValueError(f"Cannot parse action:\n{action_text}")

    species = m.group("species").strip()

    try:
        Element(species)
    except Exception as e:
        raise ValueError(f"Invalid chemical element symbol: {species}") from e

    return {
        "wyck_id": int(m.group("wyck_id")),
        "mult": int(m.group("mult")),
        "wy": m.group("wy"),
        "sym": m.group("sym"),
        "species": species,
    }


def normalize_generated_action(text: str) -> str:
    text = str(text).strip()

    # Prefer explicit ACTION block if model generated it.
    m = re.search(r"<ACTION>.*?(?:<\\ACTION>|</ACTION>)", text, flags=re.S)
    if m:
        return m.group(0).strip()

    # Otherwise extract the first Wyckoff action line and wrap it.
    m = re.search(
        r"<WYCK_id=\d+\s+mult=\d+\s+wy=\"[^\"]+\"\s+sym=\"[^\"]+\"\s+species=\"[^\"]+\">",
        text,
        flags=re.S,
    )
    if m:
        return "<ACTION>\n" + m.group(0).strip() + "\n<\\ACTION>"

    raise ValueError(f"No valid action found in generated text:\n{text}")


def canonical_action_key(action_text: str) -> str:
    action = parse_action(action_text)
    return (
        f'{action["wyck_id"]}|{action["mult"]}|{action["wy"]}|'
        f'{action["sym"]}|{action["species"]}'
    )


def apply_action_to_original(
    original_structure,
    action_text,
    symprec=1e-2,
    angle_tolerance=5.0,
):
    action = parse_action(action_text)

    blocks = extract_wyckoff_blocks(
        original_structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    wyck_id = action["wyck_id"]

    if wyck_id < 0 or wyck_id >= len(blocks):
        raise IndexError(
            f"WYCK_id={wyck_id} out of range. Extracted {len(blocks)} blocks."
        )

    block = blocks[wyck_id]
    reconstructed = original_structure.copy()

    for idx in block.indices:
        reconstructed.replace(int(idx), action["species"])

    block_check = {
        "action_wyck_id": action["wyck_id"],
        "action_mult": action["mult"],
        "action_wy": action["wy"],
        "action_sym": action["sym"],
        "action_new_species": action["species"],
        "extracted_wyck_id": block.wyck_id,
        "extracted_mult": block.mult,
        "extracted_wy": block.wy,
        "extracted_sym": block.sym,
        "extracted_old_species": block.species,
        "extracted_indices": block.indices,
        "metadata_consistent": (
            action["mult"] == block.mult
            and action["wy"] == block.wy
            and action["sym"] == block.sym
        ),
    }

    return reconstructed, action, block, block_check


# ============================================================
# Model
# ============================================================

def get_torch_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_lora_model_and_tokenizer(args):
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    #if not os.path.isdir(args.lora_path):
        #raise FileNotFoundError(f"lora_path not found: {args.lora_path}")

    if args.lora_path is not None and not os.path.isdir(args.lora_path):
        raise FileNotFoundError(f"lora_path not found: {args.lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.use_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig is not available. Please install/update transformers.")

        compute_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

        model_kwargs["quantization_config"] = bnb_config

        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": get_local_rank()}
    else:
        model_kwargs["torch_dtype"] = get_torch_dtype(args.precision)

    base = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    #model = PeftModel.from_pretrained(base, args.lora_path)
    # 如果 lora_path 存在且不为空，才加载 LoRA
    if args.lora_path and os.path.isdir(args.lora_path):
        model = PeftModel.from_pretrained(base, args.lora_path)
    else:
        model = base  # 直接使用基础模型，不加载 LoRA
    model.eval()

    if not args.use_4bit:
        device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model, tokenizer


@torch.no_grad()
def generate_actions(
    model,
    tokenizer,
    wyck_seq: str,
    current_mag_density: float,
    target_mag_density: float,
    args,
) -> List[Dict[str, Any]]:
    user_prompt = build_user_prompt(
        original_wyck_seq=wyck_seq,
        original_mag_density=f"{float(current_mag_density):.4f}",
        candidate_mag_density=f"{float(target_mag_density):.3f}",
    )

    prompt_text = build_prompt_text(tokenizer, user_prompt)

    device = model.device
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=False,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    do_sample = args.temperature > 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        num_return_sequences=args.k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    actions = []
    seen = set()

    for i, out in enumerate(outputs):
        gen_ids = out[input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        record = {
            "raw_text": gen_text,
            "valid": False,
            "action_text": None,
            "error": None,
        }

        try:
            action_text = normalize_generated_action(gen_text)
            key = canonical_action_key(action_text)
            if key in seen:
                record["error"] = "duplicate_action"
            else:
                seen.add(key)
                record["valid"] = True
                record["action_text"] = action_text
        except Exception as e:
            record["error"] = str(e)

        actions.append(record)

    return actions


# ============================================================
# MatterSim wrapper
# ============================================================

def load_mattersim_calculator(args):
    """
    Tries several MatterSim import/API styles.

    If your local MatterSim API is different, usually only this function needs editing.
    """
    Calculator = None

    import_errors = []

    try:
        from mattersim.forcefield import MatterSimCalculator as Calculator
    except Exception as e:
        import_errors.append(f"from mattersim.forcefield import MatterSimCalculator failed: {e}")

    if Calculator is None:
        try:
            from mattersim.forcefield.potential import MatterSimCalculator as Calculator
        except Exception as e:
            import_errors.append(f"from mattersim.forcefield.potential import MatterSimCalculator failed: {e}")

    if Calculator is None:
        raise ImportError(
            "Cannot import MatterSimCalculator.\n"
            + "\n".join(import_errors)
        )

    device = args.mlff_device

    if args.mattersim_checkpoint:
        attempts = [
            {"load_path": args.mattersim_checkpoint, "device": device},
            {"model_path": args.mattersim_checkpoint, "device": device},
            {"checkpoint": args.mattersim_checkpoint, "device": device},
        ]

        last_err = None
        for kwargs in attempts:
            try:
                return Calculator(**kwargs)
            except TypeError as e:
                last_err = e
                continue

        raise RuntimeError(
            "MatterSimCalculator import succeeded, but initialization with checkpoint failed. "
            f"checkpoint={args.mattersim_checkpoint}, last_error={last_err}"
        )

    # If no checkpoint provided, rely on local default.
    try:
        return Calculator(device=device)
    except TypeError:
        return Calculator()


class MatterSimRunner:
    def __init__(self, args):
        self.args = args
        self.calc = load_mattersim_calculator(args)

    def energy_per_atom(self, structure: Structure) -> float:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.calc
        e = atoms.get_potential_energy()
        return float(e) / max(len(atoms), 1)

    def relax(self, structure: Structure) -> Tuple[Structure, float, Dict[str, Any]]:
        from ase.optimize import FIRE

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.calc

        info = {
            "relax_steps": self.args.relax_steps,
            "fmax": self.args.fmax,
            "relax_cell": self.args.relax_cell,
            "optimizer": "FIRE",
        }

        if self.args.relax_steps > 0:
            target = atoms

            if self.args.relax_cell:
                try:
                    from ase.filters import FrechetCellFilter
                    target = FrechetCellFilter(atoms)
                    info["cell_filter"] = "FrechetCellFilter"
                except Exception:
                    try:
                        from ase.filters import UnitCellFilter
                        target = UnitCellFilter(atoms)
                        info["cell_filter"] = "UnitCellFilter"
                    except Exception as e:
                        info["cell_filter"] = None
                        info["cell_filter_error"] = str(e)
                        target = atoms

            opt = FIRE(target, logfile=None)
            opt.run(fmax=self.args.fmax, steps=self.args.relax_steps)

        epa = float(atoms.get_potential_energy()) / max(len(atoms), 1)
        relaxed_structure = AseAtomsAdaptor.get_structure(atoms)

        return relaxed_structure, epa, info


# ============================================================
# CHGNet wrapper
# ============================================================

class CHGNetMagPredictor:
    def __init__(self, args):
        self.args = args

        try:
            from chgnet.model.model import CHGNet
        except Exception:
            from chgnet.model import CHGNet

        if args.chgnet_checkpoint:
            try:
                self.model = CHGNet.from_file(args.chgnet_checkpoint)
            except Exception:
                self.model = CHGNet.load(args.chgnet_checkpoint)
        else:
            try:
                self.model = CHGNet.load()
            except Exception:
                self.model = CHGNet.from_file()

        if torch.cuda.is_available() and args.chgnet_device.startswith("cuda"):
            self.model.to(args.chgnet_device)

        self.model.eval()

    def predict(self, structure: Structure) -> Dict[str, Any]:
        with torch.enable_grad():
            pred = self.model.predict_structure(structure)

        # CHGNet versions may return slightly different keys.
        m = None
        for key in ["m", "magmom", "magmoms", "magmom_per_site"]:
            if isinstance(pred, dict) and key in pred:
                m = pred[key]
                break

        if m is None:
            magmoms = []
        else:
            try:
                magmoms = np.array(m, dtype=float).reshape(-1).tolist()
            except Exception:
                magmoms = []

        if len(magmoms) == 0:
            total_magmom = float("nan")
            mag_density = float("nan")
        else:
            arr = np.array(magmoms, dtype=float)
            if self.args.mag_density_mode == "sum":
                total_magmom = float(np.sum(arr))
            elif self.args.mag_density_mode == "abs_sum":
                total_magmom = float(np.sum(np.abs(arr)))
            else:
                raise ValueError(f"Unknown mag_density_mode: {self.args.mag_density_mode}")

            mag_density = total_magmom / max(float(structure.volume), 1e-12)

        out = {
            "magmoms": magmoms,
            "total_magmom": total_magmom,
            "volume": float(structure.volume),
            "mag_density": mag_density,
        }

        for key in ["e", "energy", "f", "forces", "s", "stress"]:
            if isinstance(pred, dict) and key in pred:
                try:
                    val = pred[key]
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    out[key] = val
                except Exception:
                    pass

        return out


# ============================================================
# IO
# ============================================================

def safe_csv_field_size_limit():
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        try:
            csv.field_size_limit(2**31 - 1)
        except Exception:
            pass


def read_csv_rows(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    safe_csv_field_size_limit()

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return rows, fieldnames


def append_csv_row(path: str, row: Dict[str, Any], fieldnames: List[str]):
    """
    Append one result row. If an older result file uses a previous schema,
    migrate it to the current header first so newly added timing columns do not
    produce a malformed CSV.
    """
    exists = os.path.exists(path)

    if exists:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_fieldnames = list(reader.fieldnames or [])
                old_rows = list(reader)

            if old_fieldnames != fieldnames:
                tmp_path = path + ".schema_migration.tmp"
                with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for old_row in old_rows:
                        writer.writerow({k: old_row.get(k, "") for k in fieldnames})
                os.replace(tmp_path, path)
        except Exception as e:
            raise RuntimeError(f"Failed to migrate result CSV schema: {path}") from e

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def structure_to_cif_text(structure: Structure) -> str:
    return str(CifWriter(structure))


def save_structure_cif(structure: Structure, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    CifWriter(structure).write_file(path)


def load_processed_indices(output_dir: str) -> set:
    """
    Resume only rows that already contain the new stage-timing fields.
    Rows produced by the pre-timing version are recomputed so the final timing
    summary is not silently filled with zeros for historical results.
    """
    processed = set()

    patterns = [
        os.path.join(output_dir, "results.csv"),
        os.path.join(output_dir, "results_rank*.csv"),
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        timing_version = str(row.get("timing_num_stages", "")).strip()
                        if row.get("sample_index") and timing_version:
                            processed.add(int(row["sample_index"]))
            except Exception:
                pass

    return processed


# ============================================================
# Stage timing
# ============================================================

# The inference path is divided into 12 measurable stages.
# Stages 1-5 are executed once per sample. Stages 6-12 are executed
# in every iterative round, with candidate-level stages repeated for each
# generated candidate that reaches the corresponding operation.
INFERENCE_STAGE_DEFINITIONS = {
    "parse_input_cif": "Parse the input CIF into a pymatgen Structure.",
    "initial_mattersim_energy": "Evaluate the initial MatterSim energy per atom.",
    "initial_chgnet_prediction": "Predict the initial magnetic density with CHGNet.",
    "initial_wyckoff_encoding": "Convert the initial structure to Wyck-SEQ.",
    "initial_cif_write": "Write the initial structure to a CIF file.",
    "action_generation": "Generate candidate edit actions with the language model.",
    "action_application": "Parse, validate, and apply an action to a Wyckoff block.",
    "candidate_relaxation": "Relax an edited candidate with MatterSim.",
    "candidate_energy_filter": "Compare the relaxed energy against the energy threshold.",
    "candidate_chgnet_prediction": "Predict candidate magnetic density with CHGNet.",
    "candidate_cif_write": "Write a retained candidate structure to a CIF file.",
    "round_selection_and_encoding": "Select the best candidate and build its next-round Wyck-SEQ.",
}

INFERENCE_STAGE_NAMES = list(INFERENCE_STAGE_DEFINITIONS.keys())


def make_timing_record(include_all_stages: bool = True) -> Dict[str, Any]:
    if include_all_stages:
        times = {stage: 0.0 for stage in INFERENCE_STAGE_NAMES}
        calls = {stage: 0 for stage in INFERENCE_STAGE_NAMES}
    else:
        times = {}
        calls = {}

    return {
        "stage_times_sec": times,
        "stage_call_counts": calls,
    }


def synchronize_for_timing(args):
    """
    GPU kernels are asynchronous. Synchronizing before and after a timed block
    makes stage durations reflect actual CUDA execution rather than enqueue time.
    This can be disabled with --no-timing_sync_cuda.
    """
    if (
        getattr(args, "timing_sync_cuda", True)
        and torch.cuda.is_available()
    ):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def add_stage_timing(record: Optional[Dict[str, Any]], stage: str, elapsed: float):
    if record is None:
        return

    times = record.setdefault("stage_times_sec", {})
    calls = record.setdefault("stage_call_counts", {})

    times[stage] = float(times.get(stage, 0.0)) + float(elapsed)
    calls[stage] = int(calls.get(stage, 0)) + 1


@contextmanager
def timed_stage(stage: str, args, *timing_records: Optional[Dict[str, Any]]):
    if stage not in INFERENCE_STAGE_DEFINITIONS:
        raise KeyError(f"Unknown timing stage: {stage}")

    synchronize_for_timing(args)
    start = time.perf_counter()

    try:
        yield
    finally:
        synchronize_for_timing(args)
        elapsed = time.perf_counter() - start
        for record in timing_records:
            add_stage_timing(record, stage, elapsed)


def summarize_timing_record(
    record: Dict[str, Any],
    wall_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    times = {
        stage: float(record.get("stage_times_sec", {}).get(stage, 0.0))
        for stage in INFERENCE_STAGE_NAMES
    }
    calls = {
        stage: int(record.get("stage_call_counts", {}).get(stage, 0))
        for stage in INFERENCE_STAGE_NAMES
    }

    avg_per_call = {
        stage: (times[stage] / calls[stage] if calls[stage] > 0 else 0.0)
        for stage in INFERENCE_STAGE_NAMES
    }

    stage_sum = float(sum(times.values()))
    out = {
        "num_defined_stages": len(INFERENCE_STAGE_NAMES),
        "stage_definitions": INFERENCE_STAGE_DEFINITIONS,
        "stage_times_sec": times,
        "stage_call_counts": calls,
        "stage_mean_per_call_sec": avg_per_call,
        "sum_stage_time_sec": stage_sum,
    }

    if wall_time_sec is not None:
        wall = float(wall_time_sec)
        out["wall_time_sec"] = wall
        # Includes Python bookkeeping, logging, object construction and other
        # operations outside explicitly timed stages. Clamp tiny negative values
        # caused by timer precision or CUDA synchronization overhead.
        out["unattributed_overhead_sec"] = max(0.0, wall - stage_sum)

    return out


def timing_result_fields(
    timing_summary: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "timing_num_stages": timing_summary.get(
            "num_defined_stages", len(INFERENCE_STAGE_NAMES)
        ),
        "timed_stage_sum_sec": timing_summary.get("sum_stage_time_sec", 0.0),
        "unattributed_overhead_sec": timing_summary.get(
            "unattributed_overhead_sec", 0.0
        ),
    }

    stage_times = timing_summary.get("stage_times_sec", {})
    stage_calls = timing_summary.get("stage_call_counts", {})

    for stage in INFERENCE_STAGE_NAMES:
        out[f"time_{stage}_sec"] = float(stage_times.get(stage, 0.0))
        out[f"calls_{stage}"] = int(stage_calls.get(stage, 0))

    return out


# ============================================================
# Per-sample inference
# ============================================================

def relative_error(pred: float, target: float, eps: float = 1e-12) -> float:
    if pred is None or math.isnan(float(pred)):
        return float("nan")
    return abs(float(pred) - float(target)) / max(abs(float(target)), eps)


def make_sample_id(row: Dict[str, Any], idx: int, id_col: Optional[str]) -> str:
    if id_col and id_col in row and str(row[id_col]).strip():
        return str(row[id_col]).strip()
    for c in ["material_id", "mp_id", "id", "name"]:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return f"sample_{idx}"


def infer_one_sample(
    idx: int,
    row: Dict[str, Any],
    sample_id: str,
    model,
    tokenizer,
    mattersim: MatterSimRunner,
    chgnet: CHGNetMagPredictor,
    args,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sample_timing = make_timing_record(include_all_stages=True)

    traj = {
        "sample_index": idx,
        "sample_id": sample_id,
        "target_mag_density": args.target_mag_density,
        "status": "running",
        "error": None,
        "rounds": [],
        "timing_stage_count": len(INFERENCE_STAGE_NAMES),
        "timing_stage_definitions": INFERENCE_STAGE_DEFINITIONS,
    }

    traj_path = os.path.join(args.output_dir, "trajectories", f"{idx}_{sample_id}.json")
    cif_dir = os.path.join(args.output_dir, "cifs", f"{idx}_{sample_id}")
    os.makedirs(cif_dir, exist_ok=True)

    try:
        with timed_stage("parse_input_cif", args, sample_timing):
            cif_text = str(row[args.cif_col])
            current_structure = parse_structure_from_cif(cif_text)
            initial_formula = current_structure.composition.reduced_formula

        with timed_stage("initial_mattersim_energy", args, sample_timing):
            initial_energy_per_atom = mattersim.energy_per_atom(current_structure)

        with timed_stage("initial_chgnet_prediction", args, sample_timing):
            initial_chg = chgnet.predict(current_structure)
            initial_mag_density = float(initial_chg["mag_density"])
            current_mag_density = (
                initial_mag_density
                if not math.isnan(initial_mag_density)
                else 0.0
            )

        with timed_stage("initial_wyckoff_encoding", args, sample_timing):
            current_wyck_seq = structure_to_wyckoff_seq(
                current_structure,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                ndigits=args.ndigits,
            )

        initial_cif_path = os.path.join(cif_dir, "round_0_initial.cif")
        with timed_stage("initial_cif_write", args, sample_timing):
            save_structure_cif(current_structure, initial_cif_path)

        best_overall = {
            "round": 0,
            "structure": current_structure,
            "energy_per_atom": initial_energy_per_atom,
            "mag_density": current_mag_density,
            "wyck_seq": current_wyck_seq,
            "cif_path": initial_cif_path,
        }

        stop_reason = None

        for round_id in range(1, args.num_rounds + 1):
            round_timing = make_timing_record(include_all_stages=True)
            round_wall_start = time.perf_counter()

            round_record = {
                "round": round_id,
                "input_mag_density": current_mag_density,
                "input_energy_per_atom": best_overall["energy_per_atom"],
                "generated": [],
                "valid_candidates": [],
                "selected": None,
                "stop_reason": None,
                "timing": None,
            }

            with timed_stage(
                "action_generation",
                args,
                sample_timing,
                round_timing,
            ):
                generated = generate_actions(
                    model=model,
                    tokenizer=tokenizer,
                    wyck_seq=current_wyck_seq,
                    current_mag_density=current_mag_density,
                    target_mag_density=args.target_mag_density,
                    args=args,
                )

            valid_candidates = []

            for gen_i, gen in enumerate(generated):
                candidate_timing = make_timing_record(include_all_stages=True)
                candidate_wall_start = time.perf_counter()

                cand_record = {
                    "gen_index": gen_i,
                    "raw_text": gen.get("raw_text"),
                    "valid_generation": gen.get("valid", False),
                    "action_text": gen.get("action_text"),
                    "generation_error": gen.get("error"),
                    "applied": False,
                    "metadata_consistent": None,
                    "filtered": None,
                    "filter_reason": None,
                    "relaxed_energy_per_atom": None,
                    "mag_density": None,
                    "total_magmom": None,
                    "formula": None,
                    "cif_path": None,
                    "error": None,
                    "timing": None,
                }

                if not gen.get("valid", False):
                    candidate_wall = time.perf_counter() - candidate_wall_start
                    cand_record["timing"] = summarize_timing_record(
                        candidate_timing,
                        wall_time_sec=candidate_wall,
                    )
                    round_record["generated"].append(cand_record)
                    continue

                try:
                    with timed_stage(
                        "action_application",
                        args,
                        sample_timing,
                        round_timing,
                        candidate_timing,
                    ):
                        cand_structure, action, block, block_check = apply_action_to_original(
                            current_structure,
                            gen["action_text"],
                            symprec=args.symprec,
                            angle_tolerance=args.angle_tolerance,
                        )

                        cand_record["applied"] = True
                        cand_record["action"] = action
                        cand_record["block_check"] = block_check
                        cand_record["metadata_consistent"] = bool(
                            block_check["metadata_consistent"]
                        )

                    if (
                        (not args.allow_inconsistent_action)
                        and (not block_check["metadata_consistent"])
                    ):
                        cand_record["filtered"] = True
                        cand_record["filter_reason"] = "metadata_inconsistent"
                        candidate_wall = time.perf_counter() - candidate_wall_start
                        cand_record["timing"] = summarize_timing_record(
                            candidate_timing,
                            wall_time_sec=candidate_wall,
                        )
                        round_record["generated"].append(cand_record)
                        continue

                    with timed_stage(
                        "candidate_relaxation",
                        args,
                        sample_timing,
                        round_timing,
                        candidate_timing,
                    ):
                        relaxed_structure, relaxed_epa, relax_info = mattersim.relax(
                            cand_structure
                        )
                        cand_record["relax_info"] = relax_info
                        cand_record["relaxed_energy_per_atom"] = relaxed_epa

                    with timed_stage(
                        "candidate_energy_filter",
                        args,
                        sample_timing,
                        round_timing,
                        candidate_timing,
                    ):
                        threshold = initial_energy_per_atom + args.energy_margin
                        energy_too_high = relaxed_epa > threshold

                    if energy_too_high:
                        cand_record["filtered"] = True
                        cand_record["filter_reason"] = (
                            f"energy_too_high: relaxed_epa={relaxed_epa:.6f} > "
                            f"initial_epa+margin={threshold:.6f}"
                        )
                        candidate_wall = time.perf_counter() - candidate_wall_start
                        cand_record["timing"] = summarize_timing_record(
                            candidate_timing,
                            wall_time_sec=candidate_wall,
                        )
                        round_record["generated"].append(cand_record)
                        continue

                    with timed_stage(
                        "candidate_chgnet_prediction",
                        args,
                        sample_timing,
                        round_timing,
                        candidate_timing,
                    ):
                        chg_pred = chgnet.predict(relaxed_structure)
                        mag_density = float(chg_pred["mag_density"])

                        cand_record["filtered"] = False
                        cand_record["filter_reason"] = None
                        cand_record["mag_density"] = mag_density
                        cand_record["total_magmom"] = chg_pred["total_magmom"]
                        cand_record["formula"] = (
                            relaxed_structure.composition.reduced_formula
                        )

                    cif_path = os.path.join(
                        cif_dir,
                        f"round_{round_id}_cand_{gen_i}.cif",
                    )
                    with timed_stage(
                        "candidate_cif_write",
                        args,
                        sample_timing,
                        round_timing,
                        candidate_timing,
                    ):
                        save_structure_cif(relaxed_structure, cif_path)
                        cand_record["cif_path"] = cif_path

                    valid_candidates.append(
                        {
                            "gen_index": gen_i,
                            "record": cand_record,
                            "structure": relaxed_structure,
                            "energy_per_atom": relaxed_epa,
                            "mag_density": mag_density,
                        }
                    )

                except Exception as e:
                    cand_record["error"] = str(e)
                    cand_record["traceback"] = traceback.format_exc()

                candidate_wall = time.perf_counter() - candidate_wall_start
                cand_record["timing"] = summarize_timing_record(
                    candidate_timing,
                    wall_time_sec=candidate_wall,
                )
                round_record["generated"].append(cand_record)

            if len(valid_candidates) == 0:
                stop_reason = "no_valid_candidate"
                round_record["stop_reason"] = stop_reason
                round_wall = time.perf_counter() - round_wall_start
                round_record["timing"] = summarize_timing_record(
                    round_timing,
                    wall_time_sec=round_wall,
                )
                traj["rounds"].append(round_record)
                break

            with timed_stage(
                "round_selection_and_encoding",
                args,
                sample_timing,
                round_timing,
            ):
                # Choose the remaining structure with the highest magnetic density.
                selected = max(valid_candidates, key=lambda x: x["mag_density"])

                selected_structure = selected["structure"]
                selected_record = selected["record"]

                current_structure = selected_structure
                current_mag_density = float(selected["mag_density"])
                current_wyck_seq = structure_to_wyckoff_seq(
                    current_structure,
                    symprec=args.symprec,
                    angle_tolerance=args.angle_tolerance,
                    ndigits=args.ndigits,
                )

                selected_cif_path = selected_record["cif_path"]

                round_record["selected"] = {
                    "gen_index": selected["gen_index"],
                    "mag_density": current_mag_density,
                    "energy_per_atom": selected["energy_per_atom"],
                    "formula": selected_structure.composition.reduced_formula,
                    "cif_path": selected_cif_path,
                }

                best_overall = {
                    "round": round_id,
                    "structure": current_structure,
                    "energy_per_atom": selected["energy_per_atom"],
                    "mag_density": current_mag_density,
                    "wyck_seq": current_wyck_seq,
                    "cif_path": selected_cif_path,
                }

            round_wall = time.perf_counter() - round_wall_start
            round_record["timing"] = summarize_timing_record(
                round_timing,
                wall_time_sec=round_wall,
            )
            traj["rounds"].append(round_record)

            if (
                args.stop_if_reach_target
                and current_mag_density >= args.target_mag_density
            ):
                stop_reason = "reach_target"
                break

        if stop_reason is None:
            stop_reason = "max_rounds_reached"

        final_mag_density = float(best_overall["mag_density"])
        rel_err = relative_error(final_mag_density, args.target_mag_density)
        total_time = time.perf_counter() - t0
        timing_summary = summarize_timing_record(
            sample_timing,
            wall_time_sec=total_time,
        )

        traj["status"] = "ok"
        traj["stop_reason"] = stop_reason
        traj["initial"] = {
            "formula": initial_formula,
            "energy_per_atom": initial_energy_per_atom,
            "mag_density": initial_mag_density,
            "total_magmom": initial_chg["total_magmom"],
            "volume": initial_chg["volume"],
        }
        traj["final"] = {
            "round": best_overall["round"],
            "formula": best_overall["structure"].composition.reduced_formula,
            "energy_per_atom": best_overall["energy_per_atom"],
            "mag_density": final_mag_density,
            "relative_error": rel_err,
            "cif_path": best_overall["cif_path"],
        }
        traj["total_time_sec"] = total_time
        traj["timing"] = timing_summary

        write_json(traj_path, traj)

        result = {
            "sample_index": idx,
            "sample_id": sample_id,
            "status": "ok",
            "stop_reason": stop_reason,
            "initial_formula": initial_formula,
            "initial_energy_per_atom": initial_energy_per_atom,
            "initial_mag_density": initial_mag_density,
            "target_mag_density": args.target_mag_density,
            "final_round": best_overall["round"],
            "final_formula": best_overall["structure"].composition.reduced_formula,
            "final_energy_per_atom": best_overall["energy_per_atom"],
            "final_mag_density": final_mag_density,
            "relative_error": rel_err,
            "final_cif_path": best_overall["cif_path"],
            "trajectory_path": traj_path,
            "total_time_sec": total_time,
            "error": "",
        }
        result.update(timing_result_fields(timing_summary))
        return result

    except Exception as e:
        total_time = time.perf_counter() - t0
        timing_summary = summarize_timing_record(
            sample_timing,
            wall_time_sec=total_time,
        )

        traj["status"] = "failed"
        traj["error"] = str(e)
        traj["traceback"] = traceback.format_exc()
        traj["total_time_sec"] = total_time
        traj["timing"] = timing_summary
        write_json(traj_path, traj)

        result = {
            "sample_index": idx,
            "sample_id": sample_id,
            "status": "failed",
            "stop_reason": "exception",
            "initial_formula": "",
            "initial_energy_per_atom": "",
            "initial_mag_density": "",
            "target_mag_density": args.target_mag_density,
            "final_round": "",
            "final_formula": "",
            "final_energy_per_atom": "",
            "final_mag_density": "",
            "relative_error": "",
            "final_cif_path": "",
            "trajectory_path": traj_path,
            "total_time_sec": total_time,
            "error": str(e),
        }
        result.update(timing_result_fields(timing_summary))
        return result


# ============================================================
# Merge / stats
# ============================================================

BASE_RESULT_FIELDS = [
    "sample_index",
    "sample_id",
    "status",
    "stop_reason",
    "initial_formula",
    "initial_energy_per_atom",
    "initial_mag_density",
    "target_mag_density",
    "final_round",
    "final_formula",
    "final_energy_per_atom",
    "final_mag_density",
    "relative_error",
    "final_cif_path",
    "trajectory_path",
    "total_time_sec",
    "error",
]

TIMING_RESULT_FIELDS = [
    "timing_num_stages",
    "timed_stage_sum_sec",
    "unattributed_overhead_sec",
]

for _stage in INFERENCE_STAGE_NAMES:
    TIMING_RESULT_FIELDS.extend(
        [
            f"time_{_stage}_sec",
            f"calls_{_stage}",
        ]
    )

RESULT_FIELDS = BASE_RESULT_FIELDS + TIMING_RESULT_FIELDS


def merge_rank_results(output_dir: str):
    rank_files = sorted(glob.glob(os.path.join(output_dir, "results_rank*.csv")))

    # Keep the latest row for a duplicated sample_index. This is important when
    # an old no-timing result is recomputed and appended during --resume.
    rows_by_index = {}

    for path in rank_files:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["sample_index"])
                previous = rows_by_index.get(idx)

                if previous is None:
                    rows_by_index[idx] = row
                    continue

                previous_has_timing = bool(
                    str(previous.get("timing_num_stages", "")).strip()
                )
                current_has_timing = bool(
                    str(row.get("timing_num_stages", "")).strip()
                )

                if current_has_timing or not previous_has_timing:
                    rows_by_index[idx] = row

    all_rows = [rows_by_index[idx] for idx in sorted(rows_by_index)]

    final_csv = os.path.join(output_dir, "results.csv")
    with open(final_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})

    return final_csv, all_rows


def safe_float_value(value) -> Optional[float]:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def describe_values(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": float("nan"),
            "variance_population": float("nan"),
            "variance_sample": float("nan"),
            "std_population": float("nan"),
            "std_sample": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }

    sample_variance = float(np.var(arr, ddof=1)) if arr.size >= 2 else float("nan")
    sample_std = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")

    return {
        "count": int(arr.size),
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "variance_population": float(np.var(arr, ddof=0)),
        "variance_sample": sample_variance,
        "std_population": float(np.std(arr, ddof=0)),
        "std_sample": sample_std,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def write_stage_timing_summary_csv(
    path: str,
    stage_stats: Dict[str, Dict[str, Any]],
):
    fields = [
        "stage_index",
        "stage",
        "description",
        "num_samples",
        "num_active_samples",
        "total_calls",
        "total_time_sec",
        "mean_per_sample_sec",
        "variance_population_per_sample",
        "variance_sample_per_sample",
        "std_population_per_sample",
        "std_sample_per_sample",
        "median_per_sample_sec",
        "min_per_sample_sec",
        "max_per_sample_sec",
        "p25_per_sample_sec",
        "p75_per_sample_sec",
        "p90_per_sample_sec",
        "p95_per_sample_sec",
        "mean_per_active_sample_sec",
        "mean_per_call_sec",
        "share_of_all_timed_stages",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for stage_index, stage in enumerate(INFERENCE_STAGE_NAMES, start=1):
            item = stage_stats[stage]
            per_sample = item["per_sample_all"]
            per_active = item["per_active_sample"]

            writer.writerow(
                {
                    "stage_index": stage_index,
                    "stage": stage,
                    "description": INFERENCE_STAGE_DEFINITIONS[stage],
                    "num_samples": per_sample["count"],
                    "num_active_samples": item["num_active_samples"],
                    "total_calls": item["total_calls"],
                    "total_time_sec": per_sample["sum"],
                    "mean_per_sample_sec": per_sample["mean"],
                    "variance_population_per_sample": per_sample[
                        "variance_population"
                    ],
                    "variance_sample_per_sample": per_sample["variance_sample"],
                    "std_population_per_sample": per_sample["std_population"],
                    "std_sample_per_sample": per_sample["std_sample"],
                    "median_per_sample_sec": per_sample["median"],
                    "min_per_sample_sec": per_sample["min"],
                    "max_per_sample_sec": per_sample["max"],
                    "p25_per_sample_sec": per_sample["p25"],
                    "p75_per_sample_sec": per_sample["p75"],
                    "p90_per_sample_sec": per_sample["p90"],
                    "p95_per_sample_sec": per_sample["p95"],
                    "mean_per_active_sample_sec": per_active["mean"],
                    "mean_per_call_sec": item["mean_per_call_sec"],
                    "share_of_all_timed_stages": item[
                        "share_of_all_timed_stages"
                    ],
                }
            )


def compute_stats(rows: List[Dict[str, Any]], output_dir: str):
    total = len(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    rel_errors = []
    final_mags = []
    times = []
    timed_sums = []
    overheads = []

    for r in ok_rows:
        value = safe_float_value(r.get("relative_error"))
        if value is not None:
            rel_errors.append(value)

        value = safe_float_value(r.get("final_mag_density"))
        if value is not None:
            final_mags.append(value)

        value = safe_float_value(r.get("total_time_sec"))
        if value is not None:
            times.append(value)

        value = safe_float_value(r.get("timed_stage_sum_sec"))
        if value is not None:
            timed_sums.append(value)

        value = safe_float_value(r.get("unattributed_overhead_sec"))
        if value is not None:
            overheads.append(value)

    stage_stats = {}
    total_timed_across_stages = 0.0

    raw_stage_data = {}
    for stage in INFERENCE_STAGE_NAMES:
        durations_all = []
        durations_active = []
        total_calls = 0

        for r in ok_rows:
            duration = safe_float_value(r.get(f"time_{stage}_sec"))
            calls_value = safe_float_value(r.get(f"calls_{stage}"))

            duration = 0.0 if duration is None else duration
            calls = 0 if calls_value is None else int(calls_value)

            durations_all.append(duration)
            total_calls += calls
            if calls > 0:
                durations_active.append(duration)

        per_sample_all = describe_values(durations_all)
        per_active_sample = describe_values(durations_active)
        total_stage_time = float(per_sample_all["sum"])
        total_timed_across_stages += total_stage_time

        raw_stage_data[stage] = {
            "description": INFERENCE_STAGE_DEFINITIONS[stage],
            "per_sample_all": per_sample_all,
            "per_active_sample": per_active_sample,
            "num_active_samples": len(durations_active),
            "total_calls": int(total_calls),
            "mean_per_call_sec": (
                total_stage_time / total_calls if total_calls > 0 else 0.0
            ),
        }

    for stage in INFERENCE_STAGE_NAMES:
        item = raw_stage_data[stage]
        stage_total = float(item["per_sample_all"]["sum"])
        item["share_of_all_timed_stages"] = (
            stage_total / total_timed_across_stages
            if total_timed_across_stages > 0
            else 0.0
        )
        stage_stats[stage] = item

    target_values = [
        value
        for value in (
            safe_float_value(r.get("target_mag_density")) for r in ok_rows
        )
        if value is not None
    ]

    stats = {
        "num_total": total,
        "num_ok": len(ok_rows),
        "num_failed": total - len(ok_rows),
        "target_mag_density": (
            target_values[0]
            if target_values and all(v == target_values[0] for v in target_values)
            else None
        ),
        "relative_error": {
            **describe_values(rel_errors),
            "lt_0p05": int(sum(e < 0.05 for e in rel_errors)),
            "lt_0p10": int(sum(e < 0.10 for e in rel_errors)),
            "lt_0p20": int(sum(e < 0.20 for e in rel_errors)),
            "lt_0p50": int(sum(e < 0.50 for e in rel_errors)),
        },
        "final_mag_density": describe_values(final_mags),
        "time_sec": describe_values(times),
        "timing_analysis": {
            "num_defined_stages": len(INFERENCE_STAGE_NAMES),
            "stage_order": INFERENCE_STAGE_NAMES,
            "stage_definitions": INFERENCE_STAGE_DEFINITIONS,
            "sample_timed_stage_sum_sec": describe_values(timed_sums),
            "sample_unattributed_overhead_sec": describe_values(overheads),
            "total_timed_stage_time_sec": total_timed_across_stages,
            "stages": stage_stats,
        },
        "stop_reasons": {},
        "status_counts": {},
    }

    for r in rows:
        status = r.get("status", "")
        reason = r.get("stop_reason", "")
        stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
        stats["stop_reasons"][reason] = stats["stop_reasons"].get(reason, 0) + 1

    timing_summary_csv = os.path.join(output_dir, "stage_timing_summary.csv")
    write_stage_timing_summary_csv(timing_summary_csv, stage_stats)
    stats["timing_analysis"]["summary_csv"] = timing_summary_csv

    stats_path = os.path.join(output_dir, "stats.json")
    write_json(stats_path, stats)

    return stats_path, stats


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    p.add_argument("--attn_implementation", type=str, default=None)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)

    # Data
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--cif_col", type=str, default="cif")
    p.add_argument("--id_col", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=None)

    # Iteration
    p.add_argument("--target_mag_density", type=float, default=0.2)
    p.add_argument("--num_rounds", type=int, default=5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--energy_margin", type=float, default=9999)
    p.add_argument("--stop_if_reach_target", action=argparse.BooleanOptionalAction, default=False)

    # Generation
    p.add_argument("--max_input_length", type=int, default=1300)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)

    # Symmetry
    p.add_argument("--symprec", type=float, default=1e-2)
    p.add_argument("--angle_tolerance", type=float, default=5.0)
    p.add_argument("--ndigits", type=int, default=3)
    p.add_argument("--allow_inconsistent_action", action="store_true")

    # MatterSim
    p.add_argument("--mattersim_checkpoint", type=str, default=None)
    p.add_argument("--mlff_device", type=str, default=None)
    p.add_argument("--relax_steps", type=int, default=100)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--relax_cell", action=argparse.BooleanOptionalAction, default=True)

    # CHGNet
    p.add_argument("--chgnet_checkpoint", type=str, default=None)
    p.add_argument("--chgnet_device", type=str, default=None)
    p.add_argument("--mag_density_mode", type=str, default="abs_sum", choices=["abs_sum", "sum"])

    # Runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--timing_sync_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize CUDA before and after each timed stage for accurate "
            "GPU timings. Disable with --no-timing_sync_cuda for lower overhead."
        ),
    )

    args = p.parse_args()

    args.model_path = os.path.abspath(args.model_path)
    #args.lora_path = os.path.abspath(args.lora_path)
    if args.lora_path is not None:
        args.lora_path = os.path.abspath(args.lora_path)
    args.input_csv = os.path.abspath(args.input_csv)
    args.output_dir = os.path.abspath(args.output_dir)

    if args.mlff_device is None:
        args.mlff_device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"

    if args.chgnet_device is None:
        args.chgnet_device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"

    return args


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    init_distributed_if_needed()
    setup_logging(args.output_dir)

    rank = get_rank()
    world_size = get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main_process():
        LOGGER.info("=" * 100)
        LOGGER.info("Tego iterative inference")
        LOGGER.info("=" * 100)
        LOGGER.info(json.dumps(vars(args), indent=2, ensure_ascii=False))
        LOGGER.info("Inference is divided into %d timed stages:", len(INFERENCE_STAGE_NAMES))
        for stage_index, stage in enumerate(INFERENCE_STAGE_NAMES, start=1):
            LOGGER.info(
                "  Stage %02d/%02d | %s | %s",
                stage_index,
                len(INFERENCE_STAGE_NAMES),
                stage,
                INFERENCE_STAGE_DEFINITIONS[stage],
            )

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"input_csv not found: {args.input_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "cifs"), exist_ok=True)

    rows, fieldnames = read_csv_rows(args.input_csv)

    if args.cif_col not in fieldnames:
        raise ValueError(f"cif_col={args.cif_col} not found. Available columns: {fieldnames}")

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    processed = load_processed_indices(args.output_dir) if args.resume else set()

    local_indices = [
        i for i in range(len(rows))
        if i % world_size == rank and i not in processed
    ]

    LOGGER.info(
        "Dataset loaded: total=%d | already_processed=%d | local_to_process=%d",
        len(rows),
        len(processed),
        len(local_indices),
    )

    LOGGER.info("Loading LoRA model...")
    model, tokenizer = load_lora_model_and_tokenizer(args)

    LOGGER.info("Loading MatterSim runner...")
    mattersim = MatterSimRunner(args)

    LOGGER.info("Loading CHGNet predictor...")
    chgnet = CHGNetMagPredictor(args)

    rank_csv = os.path.join(args.output_dir, f"results_rank{rank}.csv")

    iterator = tqdm(
        local_indices,
        desc=f"Rank {rank} inference",
        disable=not is_main_process(),
    )

    for idx in iterator:
        row = rows[idx]
        sample_id = make_sample_id(row, idx, args.id_col)

        LOGGER.info("Start sample idx=%d sample_id=%s", idx, sample_id)

        result = infer_one_sample(
            idx=idx,
            row=row,
            sample_id=sample_id,
            model=model,
            tokenizer=tokenizer,
            mattersim=mattersim,
            chgnet=chgnet,
            args=args,
        )

        append_csv_row(rank_csv, result, RESULT_FIELDS)

        LOGGER.info(
            "Done sample idx=%d status=%s final_mag=%s rel_error=%s "
            "total=%.2fs timed=%.2fs overhead=%.2fs",
            idx,
            result.get("status"),
            result.get("final_mag_density"),
            result.get("relative_error"),
            float(result.get("total_time_sec") or 0.0),
            float(result.get("timed_stage_sum_sec") or 0.0),
            float(result.get("unattributed_overhead_sec") or 0.0),
        )

    barrier_if_needed()

    if is_main_process():
        final_csv, all_rows = merge_rank_results(args.output_dir)
        stats_path, stats = compute_stats(all_rows, args.output_dir)

        LOGGER.info("Merged results saved to: %s", final_csv)
        LOGGER.info("Stats saved to: %s", stats_path)
        LOGGER.info(json.dumps(stats, indent=2, ensure_ascii=False))

    barrier_if_needed()


if __name__ == "__main__":
    main()