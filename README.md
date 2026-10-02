# Reproducibility code for high-dimensional saddlepoint approximation

This repository contains the Python code used to reproduce the numerical experiments in Section 8 of

**A. Katsevich, “Saddlepoint Approximation and Central Limit Theorem for Densities in High Dimensions.”**

The equation and figure cross-references below correspond to the manuscript version

[arXiv:2510.21545](https://arxiv.org/abs/2510.21545).

## Requirements

Python 3.7+ and the following packages:

- NumPy
- SciPy
- Matplotlib

No automatic-differentiation package is required. The derivatives used in Appendix C are evaluated analytically.

## Running the experiment

On Windows, double-click

`run_section8.bat`

or run from a terminal:

```bash
python section8_reproduce_all.py
```

The Windows batch file checks for the required Python packages and attempts to install any that are missing.

## Output

The script creates

```text
Section8_reproduction/
    Figures/
    data/
```

next to the script. If the output folder already exists, the script reuses it and overwrites the current output files. It does not delete the folder or remove unrelated files.

### Figures

The script generates the six PNG source images used in the paper:

- `mu.png`
- `log_errors_r_1_6.png`
- `log_errors_r_1_3.png`
- `log_errors_r_1.png`
- `log_errors_r_4_3.png`
- `reconstruction_comparison.png`

These correspond to the four numbered figures in the paper:

- **Figure 1:** `mu.png`
- **Figure 2:** `log_errors_r_1_6.png` (top), `log_errors_r_1_3.png` (bottom)
- **Figure 3:** `log_errors_r_1.png` (top), `log_errors_r_4_3.png` (bottom)
- **Figure 4:** `reconstruction_comparison.png`

Thus there are six PNG assets but four numbered figures, because Figures 2 and 3 each contain two separately generated profile panels. The script does not generate combined profile figures or duplicate PDF figure files.

The horizontal axis of the four profile plots is the angular coordinate \(\xi\) defined in equation (8.37).

### Numerical outputs

The `data` directory contains CSV, JSON, and text files with the numerical values used to reproduce or check:

- Table 1 and the standardized mode quantities in equation (8.28);
- \(c_{3,0}\), \(c_{4,0}\), the explicit Theorem 3.1 bound at \(a=0\), and the exact central SPA error in equations (8.29)–(8.31);
- the Monte Carlo estimate of \(J(0)\) and its comparison with the exact \(I(0)\) in equations (8.32)–(8.33);
- the shell profiles in Figures 2–3, based on equations (8.34)–(8.37);
- the true-image normalization, relative reconstruction errors, and reconstruction definitions in equations (8.39)–(8.41).

These files are numerical reproducibility outputs, not additional figures.

## Reproducibility details

The script is intentionally heavily commented and cross-referenced to the paper, including Appendices B and C. In particular:

1. Every random-number generator uses an explicit fixed seed.
2. The exact finite Gaussian mixture in equation (8.14) is evaluated in log-space using `scipy.special.logsumexp`.
3. The same data realization is used for every reconstruction in Figure 4.
4. Figure 4 uses a common color scale for all displayed images.
5. The conventional SPA is evaluated from equation (8.16), equivalently from equation (2.22) with \(I(a)=1\). For this two-component Gaussian mixture, the \(d\)-dimensional saddlepoint equation reduces exactly to a single scalar root; the derivation is documented in the code.
6. For Figures 2–3, rotational symmetry reduces the density evaluation to the two-dimensional plane spanned by the distinguished direction \(e\) and one transverse direction. No random directions are used.
7. The Monte Carlo calculation in equations (8.32)–(8.33) is realization-dependent, so its random seed is fixed to make the run exactly repeatable.
8. No numerical values reported in the paper are hard-coded as outputs; they are recomputed from the model parameters and fixed seeds.

## Repository files

- `section8_reproduce_all.py` — complete reproduction script
- `run_section8.bat` — Windows launcher
- `README.md` — this file
