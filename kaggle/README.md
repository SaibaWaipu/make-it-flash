# Kaggle

This directory contains a self-contained notebook generated from the tracked package sources.

1. Regenerate after source edits: python scripts/build_kaggle_notebook.py (from the project directory).
2. From the project directory, push the private kernel with: kaggle kernels push -p kaggle.
3. The notebook requests GPU + Internet, prepares the calibration sample, then starts teacher caching only if at least 66 GiB is free.

Kaggle GPU types vary. If preflight rejects the available GPU, no 32B teacher weights have been loaded; use a compatible HF Jobs flavor after checking its current price, or defer the teacher pass.
