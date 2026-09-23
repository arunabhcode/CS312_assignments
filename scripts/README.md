# Student Scripts

One-time helper scripts for setting up Modal and understanding the codebase.

Run these from the repo root after filling in the config block in `utils.py`,
using module syntax so Python can import the repo's top-level modules:

```bash
uv run python -m scripts.modal_usage
```

## Files

- `modal_usage.py`: prints the current Modal billing summary for an environment.
- `inspect_data_row.py`: launches a small Modal job in your configured
  environment, reads one hardcoded training row from the binary DCLM data, and
  decodes it with the course SentencePiece tokenizer.

```bash
uv run python -m scripts.inspect_data_row
```

Course staff manage environment provisioning and assignment budgets separately.
