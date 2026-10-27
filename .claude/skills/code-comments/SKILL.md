---
name: code-comments
description: Comment and docstring style for the TAMPanda codebase. Use when writing or editing Python code in this repo (tampanda/, examples/, tests/), and when asked to clean up or review comments.
---

# Comments and docstrings in TAMPanda

Match the existing code (e.g. `rrt_star.py`, `mink_ik.py`, `pick_place.py`):
few comments, each one earning its place.

## Rules

1. **Comment the why, never the what.** If a line only restates the code
   (`# compute q4`, `# return the result`), delete it. Keep comments for
   non-obvious reasons, constraints, units/frames, and workarounds.
2. **Module docstring: one line**, plus at most a short paragraph when the
   module needs context (an external reference, a licence note, a frame
   convention). No step-by-step algorithm walkthroughs.
3. **Docstrings only on public, non-obvious functions and classes.**
   Private helpers and self-explanatory methods get none, or a single line.
4. **Args sections only for non-obvious parameters.** Skip parameters whose
   name and type already say it (`approach_dist`, `table_z`, `seed`). One
   line per entry.
5. **Cite once.** A paper or upstream project is named once, where the
   idea is used, as `Author et al., Venue Year`. No repeated citations, no
   abstracts.
6. **No history or debugging narratives** ("tightened from 0.007",
   "previously this failed because...", "kept for compatibility" essays).
   That belongs in commit messages or PR descriptions.
7. **No section banners** (`# ===== ... =====`) unless a file holds two
   clearly separate parts; then a single short `# --- name ---` line.
8. **Inline comments are short** (fit on the line or one line above) and in
   English, like the rest of the code.
9. **Tests:** a one-line docstring saying what behaviour is checked; explain
   setup only when it is not obvious from the code.

## Quick check before finishing

- Would deleting this comment lose information the code does not show? If
  not, delete it.
- Is any docstring longer than the function it documents? Shorten it.
- Does the file's comment density look like its neighbours? Aim for that.
