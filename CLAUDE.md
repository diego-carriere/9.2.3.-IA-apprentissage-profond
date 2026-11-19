# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Never add Claude as co-author in commits (no `Co-Authored-By: Claude ...` trailer), and no "Generated with Claude Code" line in PRs.

## Context

Deep learning course project (EMA, module 9.2.3). The repo is at an early stage: `src/` (code) and `models/` (saved weights) are empty scaffolding.

The parent directory (outside the repo) holds course material: lecture PDFs, `TD1.pdf`, and `TP1/TP1.ipynb` with a saved `model.pth` — useful for the PyTorch conventions the course uses.

## Environment

The virtualenv lives outside the repo, at `../.venv` (Python 3.14, CPU-only PyTorch, plus torchvision, gymnasium, scikit-learn, matplotlib, ipykernel).

```bash
source ../.venv/bin/activate
pip install -r requirements.txt   # includes the PyTorch CPU wheel index
```

After adding a dependency, regenerate `requirements.txt` keeping the index line:

```bash
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; ../.venv/bin/pip freeze; } > requirements.txt
```

No build, lint or test tooling is configured yet.
