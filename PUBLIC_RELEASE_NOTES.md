# Public release notes

This public-facing package makes only small organizational changes to the supplied
research scripts.

## File-name mapping

- `train_v2.py` → `train.py`
- `infer_with_stage_timing.py` → `infer.py`

## Code-level changes

- Removed private server paths from top-of-file usage examples.
- Changed the default training output directory from a private absolute path to
  `outputs/train`.
- Simplified logger names and public-facing banner text.
- Simplified one local-model-path error message.
- Kept the core model, data loading, Wyckoff processing, training, candidate
  generation, MatterSim, CHGNet, filtering, iteration, distributed execution, and
  timing logic unchanged.

## Repository-level additions

- `README.md`
- `requirements.txt`
- `environment.yml`
- `.gitignore`
- `data/README.md`
- `scripts/train.sh`
- `scripts/infer.sh`
