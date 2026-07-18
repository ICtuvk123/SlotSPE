# Contributing to SlotSPE

## One-time setup

1. Clone the repository and create a Python 3.10 environment as described in
   [README.md](README.md).
2. Install PyTorch for the local CUDA platform, then run
   `pip install -r requirements.txt`.
3. Download RNA tables and WSI features separately. Do not commit datasets,
   model weights, checkpoints, access tokens, or patient-level artifacts.
4. Verify the checkout with:

   ```bash
   python -m unittest discover -s tests -v
   ```

## Branch and review workflow

- Keep `main` stable and create one branch per change:

  ```bash
  git switch main
  git pull --ff-only
  git switch -c <username>/<short-topic>
  ```

- Commit focused changes with an explanatory message. Before pushing, inspect
  both `git status` and `git diff --cached` so generated or restricted files do
  not enter the commit.
- Push the branch and open a pull request into `main`. At least one collaborator
  should review changes that affect data alignment, survival metrics, event
  grounding, or experiment protocols.
- Do not force-push shared branches. Bring `main` into a feature branch with a
  regular merge or a carefully coordinated rebase.

## Reproducible experiments

Pull requests that change experiments should record:

- the dataset/study and split;
- the feature encoder and feature dimension;
- the random seed and important hyperparameters;
- the exact command or launcher;
- validation metrics and the output location (without committing outputs).

Generated outputs belong in the ignored `results*`, `hypersearch/runs`, or
`logs` directories. Small, non-sensitive summaries may be added deliberately
after review.

## Security and data policy

- Supply `HF_TOKEN`, `LLM_API_KEY`, and similar credentials only through local
  environment variables or an ignored `.env` file.
- Never commit raw WSIs, restricted TCGA/GDC downloads, identifiable clinical
  data, RNA matrices, gated CONCH weights, or training checkpoints.
- If a secret is committed, revoke it immediately and notify the repository
  owner; deleting it in a later commit is not sufficient.
