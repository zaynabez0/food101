# CLAUDE.md

## Project
Reproduction in PyTorch of Gallo et al. (IVCNZ 2020), "Image and Text fusion for UPMC Food-101
using BERT and CNNs". Image model = InceptionV3, text model = BERT + LSTM, then late fusion
and early fusion (stacking). GPU training runs on Modal. Data comes from Kaggle via the API.

## How to work with me
- You write the code. I want to understand it, so for every change:
  1. Before coding: say in 2–4 sentences what you are about to do and why.
  2. After coding: explain the key parts (tensor shapes, design choices, anything non-obvious).
  3. When something follows the paper, cite the table/section (e.g. "Table II"). When the paper
     is ambiguous and you made a choice, say so explicitly and flag it with `# CHOICE:` in the code.
- Work in small steps (one module or feature at a time). Don't write the whole project at once.
- If I ask "why?", answer the concept first, then the code.

## Code style
- Clean and readable over clever. Short functions, descriptive names, no deep nesting.
- Type hints on all function signatures. Short docstrings saying what goes in and out.
- Comment tensor shapes on important lines, e.g. `# (B, L, 768)`.
- No magic numbers in code: hyperparameters live in `configs/*.yaml`.
- Keep model definitions (`src/models/`), data (`src/data.py`), training (`src/train.py`)
  and Modal plumbing (`modal_app.py`) separate.
- Minimal dependencies: torch, torchvision, transformers, pandas, scikit-learn, pyyaml, kaggle, modal.

## Rules
- Use a validation split carved from train for all decisions (LR drops, early stopping).
  The test set is only used for final reported numbers.
- Never commit data, checkpoints, `.pt` files, `kaggle.json` or any secret.
- Set seeds and log every run's config + metrics to `results/`.
