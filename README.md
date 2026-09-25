# Reproducibility code for high-dimensional saddlepoint approximation

This repository contains the Python code used to reproduce the numerical
experiments in Section 8 of

**A. Katsevich, "Saddlepoint Approximation and Central Limit Theorem for
Densities in High Dimensions."**

## Requirements

The code requires Python and the following packages:

- NumPy
- SciPy
- Matplotlib

## Running the experiment

On Windows, double-click

`run_section8_v3.bat`

Alternatively, run

```bash
python section8_reproduce_all.py
```

The script creates a folder `Section8_reproduction` containing the figures,
tables, and numerical data reported in the paper.

## Reproducibility

All random-number generators use fixed seeds. The script recomputes the
reported numerical quantities from the model parameters; the reported
results are not hard-coded.
