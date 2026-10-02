#!/usr/bin/env python3
r"""
Complete reproducibility code for Section 8 of

    A. Katsevich,
    "SADDLEPOINT APPROXIMATION AND CENTRAL LIMIT THEOREM
     FOR DENSITIES IN HIGH DIMENSIONS"

See arXiv:2510.21545.
  

PURPOSE
-------
This script reproduces the complete numerical experiment in the CURRENT
version of Section 8, including

  * Figure 1: the fixed smooth mixture-center pattern mu;
  * Table 1: exact / conventional-SPA / cubic-SPA / Gaussian mode locations;
  * equation (8.28): the standardized mode locations and their errors;
  * equations (8.29)--(8.31): c_{3,0}, c_{4,0}, the explicit part of the
    Theorem 3.1 bound at a=0, and the exact central SPA error;
  * equations (8.32)--(8.33): the M=10^7 Monte Carlo estimate of J(0) and its
    comparison with the exact I(0);
  * Figure 2: signed log-density errors on S_{1/6} and S_{1/3}; the two
    profile panels are saved as separate image files, exactly as included in
    the manuscript;
  * Figure 3: signed log-density errors on S_1 and S_{4/3}; again the two
    profile panels are saved separately;
  * Figure 4: true image, averaged noisy image, exact-likelihood reconstruction,
    and conventional-SPA reconstruction;
  * equations (8.39)--(8.41): the true-image normalization, the relative L2
    errors, and the definitions of the four reconstructed images.

The x-axis in Figures 2--3 uses the CURRENT notation xi from equation (8.37),
not the older symbol s.

The code is intentionally verbose and heavily cross-referenced.  The main
paper formulas used are (2.21)--(2.22), (3.1)--(3.2), (4.2), (4.21),
(8.1)--(8.41), together with Appendices B and C.

DEPENDENCIES
------------
Python 3.7+ with

    numpy, scipy, matplotlib

No automatic-differentiation package is required.  The derivatives used in
Appendix C are evaluated analytically.

RUN
---

    python section8_reproduce_all.py

The script creates

    Section8_reproduction/
        Figures/
        data/

next to this file.  The script generates the six PNG source images used by the
manuscript: ``mu.png``, the four shell-profile PNGs, and
``reconstruction_comparison.png``.  The four shell profiles are separate files;
LaTeX groups the first two into Figure 2 and the last two into Figure 3.  Existing
unrelated files in the output directories are left untouched.  The ``data``
directory contains CSV files and a human-readable numerical summary.

REPRODUCIBILITY CONVENTIONS
---------------------------
1. Every random-number generator has an explicit frozen seed.
2. The exact finite Gaussian mixture (8.14) is evaluated in log-space using
   scipy.special.logsumexp.
3. The same data realization is used for every reconstruction in Figure 4.
4. Figure 4 uses a common color scale for all four displayed images.
5. The conventional SPA is evaluated from (8.16), equivalently from (2.22)
   with I(a)=1.  For this special two-component Gaussian mixture, its
   d-dimensional saddlepoint equation is reduced exactly to ONE scalar root;
   the derivation is documented below.
6. For Figures 2--3, rotational symmetry reduces the density evaluation to the
   two-dimensional plane spanned by e and one transverse direction.  No random
   directions are needed.  The four profiles are saved individually using the
   filenames appearing in the manuscript: log_errors_r_1_6, log_errors_r_1_3,
   log_errors_r_1, and log_errors_r_4_3.
7. The Monte Carlo calculation in (8.32)--(8.33) is realization-dependent.
   The seed is frozen below so the run is exactly repeatable.
8. No numerical values reported in the paper are hard-coded as outputs.  They
   are recomputed from the model, parameters, and frozen seeds.
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize, minimize_scalar, root
from scipy.special import gammaln, logsumexp


# =============================================================================
# 0. OUTPUT LOCATIONS, PAPER PARAMETERS, AND FROZEN RANDOM SEEDS
# =============================================================================

START_TIME = time.time()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Section8_reproduction"
FIG = OUT / "Figures"
DATA = OUT / "data"

# Create the output directories if they do not already exist.  Existing current
# output files are simply overwritten when the script saves them below.
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# Equation (8.23): numerical regime used in Section 8.3.
m = 6
d = m * m                 # ambient image dimension d=m^2=36
n = 1000
alpha = 0.1
sigma = 0.25              # current paper notation in (8.5), not epsilon
c_minus = 0.7
c_plus = 1.9

# Equation (8.24): m_theta=B theta with 4x4=16 DCT coefficients.
p1 = 4
p = p1 * p1

# Equation (8.4): the two-point latent state S.
p_high = 0.25
p_low = 0.75
s_minus = -1.0 / np.sqrt(3.0)
s_plus = np.sqrt(3.0)

# Equation (3.1): radius R=2.5 in the local Fourier ball.
R_LOCAL = 2.5

# Equations (8.34)--(8.37): shells and angular coordinate used in Figures 2--3.
SHELL_RADII = (1.0 / 6.0, 1.0 / 3.0, 1.0, 4.0 / 3.0)
XI_GRID = np.linspace(-1.0, 1.0, 401)

# Frozen random seeds.
# SEED_MU fixes the smooth pattern mu shown in Figure 1.
# SEED_DATA fixes the one 1000-exposure realization used in Figure 4.
# SEED_MC fixes the Monte Carlo value in (8.32).
# SEED_C3C4 fixes the numerical compact optimization in Appendix C.
SEED_MU = 20260910
SEED_DATA = 777
SEED_MC = 81
SEED_C3C4 = 1234

# Equation (8.32): M=10^7 Monte Carlo samples.
MC_SAMPLES = 10_000_000
MC_CHUNK = 250_000

# Numerical tolerances.
OPT_OPTIONS = {"maxiter": 300, "ftol": 1.0e-13, "gtol": 1.0e-9}
SCALAR_ROOT_EPS = 1.0e-14


# =============================================================================
# 1. THE 4x4 DCT IMAGE MODEL AND THE CROSS-SHAPED TRUTH: (8.24), (8.39)
# =============================================================================


def make_1d_dct_basis(size: int) -> np.ndarray:
    """Return the orthonormal DCT-II basis as ROWS.

    For x=0,...,size-1 and k=0,...,size-1,

        b_0(x) = sqrt(1/size),
        b_k(x) = sqrt(2/size) cos(pi (x+1/2) k / size),  k>0.

    Tensor products of the first four one-dimensional modes give the columns of
    B in equation (8.24).
    """
    x = np.arange(size)
    rows = []
    for k in range(size):
        v = np.cos(np.pi * (x + 0.5) * k / size)
        v *= np.sqrt(1.0 / size) if k == 0 else np.sqrt(2.0 / size)
        rows.append(v)
    return np.asarray(rows)


DCT1 = make_1d_dct_basis(m)

# Equation (8.24): B:R^16 -> R^36 consists of the first 4x4 tensor-product
# orthonormal DCT modes. ``kl[j]`` records the two frequencies of column j.
modes: List[np.ndarray] = []
kl: List[Tuple[int, int]] = []
for k in range(p1):
    for ell in range(p1):
        modes.append(np.outer(DCT1[k], DCT1[ell]).reshape(-1))
        kl.append((k, ell))
B = np.column_stack(modes)
assert np.allclose(B.T @ B, np.eye(p), atol=1.0e-13)

# Equation (8.39): the true image starts from a fixed cross-shaped template,
# is projected into the DCT model space, and is then normalized so that
# max_x |m_{theta0}(x)| = 1.5/sqrt(n).
cross_template = np.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.08, 0.18, 0.00, 0.00],
        [0.00, 0.18, 0.45, 0.68, 0.18, 0.00],
        [0.00, 0.28, 0.68, 1.00, 0.55, 0.00],
        [0.00, 0.00, 0.18, 0.55, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)

cross_vec = cross_template.reshape(-1)
theta_cross_projection = B.T @ cross_vec
raw_true_image = B @ theta_cross_projection
truth_scale = (1.5 / np.sqrt(n)) / np.max(np.abs(raw_true_image))
theta0 = truth_scale * theta_cross_projection
m_true = B @ theta0
assert np.isclose(np.max(np.abs(m_true)), 1.5 / np.sqrt(n), rtol=1.0e-13)


# =============================================================================
# 2. PERIODIC CORRELATION OPERATOR AND MIXTURE-CENTER PATTERN: (8.2)--(8.9)
# =============================================================================

# Equation (8.2): periodic five-point stencil.  c_alpha normalizes the squared
# row norm of G_alpha.
c_alpha = 1.0 / np.sqrt(1.0 + 4.0 * alpha**2)
G_alpha = np.zeros((d, d))


def pixel_index(i: int, j: int) -> int:
    """Periodic row-major index on the m x m pixel torus."""
    return (i % m) * m + (j % m)


for i in range(m):
    for j in range(m):
        row = pixel_index(i, j)
        G_alpha[row, row] += c_alpha
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            G_alpha[row, pixel_index(i + di, j + dj)] += c_alpha * alpha

C_alpha = G_alpha @ G_alpha.T
C_alpha_inv = np.linalg.inv(C_alpha)
assert np.min(np.linalg.eigvalsh(C_alpha)) > 0.0  # equation (8.3), |alpha|<1/4

# Section 8.3 / Figure 1: mu lies in the same 4x4 DCT subspace.  The 16 DCT
# coefficients are one frozen N(0,1) draw multiplied by weights
# (1+k+ell)^(-1/2), then normalized to ||mu||=1.5.
rng_mu = np.random.default_rng(SEED_MU)
mu_frequency_weights = np.array([(1.0 + k + ell) ** (-0.5) for k, ell in kl])
mu_coefficients = rng_mu.normal(size=p) * mu_frequency_weights
mu = B @ mu_coefficients
mu *= 1.5 / np.linalg.norm(mu)
assert np.isclose(np.linalg.norm(mu), 1.5, rtol=1.0e-13)

# Equations (8.4), (8.7), (8.15).
mean_S = p_low * s_minus + p_high * s_plus
second_moment_S = p_low * s_minus**2 + p_high * s_plus**2
c_bar = p_low * c_minus + p_high * c_plus
kappa3 = p_low * s_minus**3 + p_high * s_plus**3
beta = p_low * s_minus * c_minus + p_high * s_plus * c_plus
assert np.isclose(mean_S, 0.0, atol=1.0e-15)
assert np.isclose(second_moment_S, 1.0, atol=1.0e-15)
assert np.isclose(c_bar, 1.0, atol=1.0e-15)
assert np.isclose(kappa3, 2.0 / np.sqrt(3.0), rtol=1.0e-14)
assert np.isclose(beta, 3.0 * np.sqrt(3.0) / 10.0, rtol=1.0e-14)

# Equation (8.8): covariance of one noise vector X_j.
Sigma = sigma**2 * (C_alpha + np.outer(mu, mu))
Sigma_inv = np.linalg.inv(Sigma)
sign_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma)
assert sign_Sigma > 0


def symmetric_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite square root."""
    eigvals, eigvecs = np.linalg.eigh(M)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def symmetric_inverse_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite inverse square root."""
    eigvals, eigvecs = np.linalg.eigh(M)
    return (eigvecs * eigvals**(-0.5)) @ eigvecs.T


Sigma_sqrt = symmetric_sqrt(Sigma)
Sigma_inv_sqrt = symmetric_inverse_sqrt(Sigma)

# Equation (8.26): distinguished axial direction e in whitened coordinates.
v_whiten = Sigma_inv_sqrt @ mu
vnorm = float(np.linalg.norm(v_whiten))
e_axis = v_whiten / vnorm

# Choose one arbitrary unit vector f_axis perpendicular to e_axis.  Rotational
# symmetry implies that any such transverse direction gives the same profiles.
j0 = int(np.argmin(np.abs(e_axis)))
f_axis = np.eye(d)[:, j0] - e_axis[j0] * e_axis
f_axis /= np.linalg.norm(f_axis)
assert abs(e_axis @ f_axis) < 1.0e-13

# Appendix C.3 consistency check:
#   sigma^2 (M C_alpha M + v v^T) = I,  M=Sigma^{-1/2}, v=M mu.
appendix_C_identity_residual = np.linalg.norm(
    sigma**2
    * (Sigma_inv_sqrt @ C_alpha @ Sigma_inv_sqrt + np.outer(v_whiten, v_whiten))
    - np.eye(d),
    ord=2,
)
assert appendix_C_identity_residual < 1.0e-11

# Figure 1.
plt.rcParams["axes.formatter.use_mathtext"] = True
fig, ax = plt.subplots(figsize=(4.4, 4.1))
h = ax.imshow(mu.reshape(m, m))
ax.set_title(r"Mixture-center pattern $\mu$")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
# Keep the manuscript/source filename unchanged.
fig.savefig(FIG / "mu.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 3. EXACT n+1-COMPONENT DENSITY / LIKELIHOOD: (8.10)--(8.14)
# =============================================================================

# K=#{j:S_j=s_+} is Bin(n,1/4), equation (8.10).
k_grid = np.arange(n + 1, dtype=float)
log_pi_k = (
    gammaln(n + 1.0)
    - gammaln(k_grid + 1.0)
    - gammaln(n - k_grid + 1.0)
    + k_grid * np.log(p_high)
    + (n - k_grid) * np.log(p_low)
)

# Equations (8.11)--(8.12): conditional mean m_k=coeff_k*mu and covariance
# v_k C_alpha for the sample average.
coeff_k = sigma / n * ((n - k_grid) * s_minus + k_grid * s_plus)
v_k = sigma**2 / n**2 * ((n - k_grid) * c_minus + k_grid * c_plus)

# Constants used in equation (8.14).
mu_Cinv_mu = mu @ C_alpha_inv @ mu
mean_quadratic_k = coeff_k**2 * mu_Cinv_mu
log_v_normalizer = -(d / 2.0) * np.log(v_k)
Cinv_mu = C_alpha_inv @ mu
sign_C, logdet_C_alpha = np.linalg.slogdet(C_alpha)
assert sign_C > 0
TRUE_CONSTANT = -(d / 2.0) * np.log(2.0 * np.pi) - 0.5 * logdet_C_alpha


def exact_logdensity_at_a(a: np.ndarray) -> float:
    """Exact log pdf log rho_n(a) of Xbar_n from (8.14)."""
    a = np.asarray(a, dtype=float)
    Cinv_a = C_alpha_inv @ a
    a_Cinv_a = a @ Cinv_a
    mu_Cinv_a = mu @ Cinv_a
    distance_sq = a_Cinv_a - 2.0 * coeff_k * mu_Cinv_a + mean_quadratic_k
    component_logs = log_pi_k + log_v_normalizer - distance_sq / (2.0 * v_k)
    return float(TRUE_CONSTANT + logsumexp(component_logs))


# =============================================================================
# 4. GAUSSIAN AND CUBIC CENTRAL APPROXIMATIONS: (8.17)--(8.20), APPENDIX B
# =============================================================================

# Equation (8.17): full density normalizing constant.
C_n_Sigma = (
    (d / 2.0) * np.log(n)
    - (d / 2.0) * np.log(2.0 * np.pi)
    - 0.5 * logdet_Sigma
)

# Equation (8.20) / Appendix B.7--B.10: contraction vector q.
tr_C_SigmaInv = np.trace(C_alpha @ Sigma_inv)
mu_SigmaInv_mu = mu @ Sigma_inv @ mu
C_SigmaInv_mu = C_alpha @ (Sigma_inv @ mu)
q_vec = sigma**3 * (
    kappa3 * mu_SigmaInv_mu * mu
    + beta * (tr_C_SigmaInv * mu + 2.0 * C_SigmaInv_mu)
)


def T3_contraction(u: np.ndarray) -> float:
    """Return T_3[u,u,u] from equation (8.20), equivalently Appendix B.5."""
    z = mu @ u
    u_C_u = u @ (C_alpha @ u)
    return float(sigma**3 * (kappa3 * z**3 + 3.0 * beta * z * u_C_u))


def gaussian_logdensity_at_a(a: np.ndarray) -> float:
    """Gaussian-order log density ell_n^G(a), equation (8.18)."""
    a = np.asarray(a, dtype=float)
    u = Sigma_inv @ a
    return float(C_n_Sigma - (n / 2.0) * (a @ u))


def cubic_logdensity_at_a(a: np.ndarray) -> float:
    """Cubic log density ell_n^C(a), equations (8.19)--(8.20)."""
    a = np.asarray(a, dtype=float)
    u = Sigma_inv @ a
    return float(
        gaussian_logdensity_at_a(a)
        + (n / 6.0) * T3_contraction(u)
        - 0.5 * (q_vec @ u)
    )


# =============================================================================
# 5. CONVENTIONAL SPA: (2.22), (8.16)
# =============================================================================

# The conventional SPA requires the saddlepoint tau solving grad phi(tau)=a.
# A generic implementation would solve a 36-dimensional nonlinear system.
# For the special two-Gaussian mixture (8.9), this can be reduced EXACTLY to a
# scalar root.
#
# Let w be the tilted probability of the s_+ state.  Put
#
#   sbar(w) = (1-w)s_- + w s_+,
#   cbar(w) = (1-w)c_- + w c_+.
#
# Differentiating (8.9) gives
#
#   grad phi(tau) = sigma sbar(w) mu + sigma^2 cbar(w) C_alpha tau.
#
# Hence, if grad phi(tau)=a,
#
#   tau(w) = C_alpha^{-1}[a-sigma sbar(w)mu]/[sigma^2 cbar(w)].
#
# The definition of w from the two mixture exponents then gives one scalar
# log-odds equation.  Solving that equation is mathematically equivalent to
# solving the full saddlepoint system and is much faster for the reconstruction.

delta_s = s_plus - s_minus
delta_c = c_plus - c_minus
log_prior_odds = np.log(p_high / p_low)


def conventional_spa_logdensity_at_a(
    a: np.ndarray,
    return_details: bool = False,
):
    """Conventional SPA log density at a, equation (8.16).

    Parameters
    ----------
    a : ndarray, shape (d,)
        Point at which rho_n^S(a) is evaluated.
    return_details : bool
        If True, also return tau, tilted high-state probability w, and H(tau).
    """
    a = np.asarray(a, dtype=float)

    def tau_from_w(w: float) -> np.ndarray:
        sbar = s_minus + w * delta_s
        cbar = c_minus + w * delta_c
        return C_alpha_inv @ (a - sigma * sbar * mu) / (sigma**2 * cbar)

    def scalar_equation(w: float) -> float:
        tau = tau_from_w(w)
        m_tau = mu @ tau
        q_tau = tau @ (C_alpha @ tau)
        rhs = (
            log_prior_odds
            + sigma * delta_s * m_tau
            + 0.5 * sigma**2 * delta_c * q_tau
        )
        return np.log(w) - np.log1p(-w) - rhs

    w = brentq(
        scalar_equation,
        SCALAR_ROOT_EPS,
        1.0 - SCALAR_ROOT_EPS,
        xtol=1.0e-14,
        rtol=1.0e-14,
        maxiter=200,
    )
    tau = tau_from_w(w)

    m_tau = mu @ tau
    q_tau = tau @ (C_alpha @ tau)
    state_logs = np.array(
        [
            np.log(p_low)
            + sigma * s_minus * m_tau
            + 0.5 * sigma**2 * c_minus * q_tau,
            np.log(p_high)
            + sigma * s_plus * m_tau
            + 0.5 * sigma**2 * c_plus * q_tau,
        ]
    )
    phi_tau = float(logsumexp(state_logs))

    # Hessian H(tau).  The first term is the tilted average of the component
    # covariance matrices; the rank-one term is the covariance of the two
    # component score vectors under the tilted state law.
    cbar = c_minus + w * delta_c
    grad_difference = sigma * delta_s * mu + sigma**2 * delta_c * (C_alpha @ tau)
    H = sigma**2 * cbar * C_alpha + w * (1.0 - w) * np.outer(
        grad_difference, grad_difference
    )
    sign_H, logdet_H = np.linalg.slogdet(H)
    if sign_H <= 0:
        raise RuntimeError("Saddlepoint Hessian is not positive definite")

    phi_star = float(tau @ a - phi_tau)
    value = (
        (d / 2.0) * np.log(n)
        - (d / 2.0) * np.log(2.0 * np.pi)
        - 0.5 * logdet_H
        - n * phi_star
    )

    # Implementation check: the full d-dimensional saddle equation must hold.
    sbar = s_minus + w * delta_s
    grad_phi = sigma * sbar * mu + sigma**2 * cbar * (C_alpha @ tau)
    saddle_residual = float(np.linalg.norm(grad_phi - a))
    if saddle_residual > 2.0e-9:
        raise RuntimeError(f"Conventional-SPA saddle residual too large: {saddle_residual}")

    if return_details:
        return float(value), tau, float(w), H, saddle_residual
    return float(value)


# At a=0, tau=0 and H(0)=Sigma, so equation (8.16) must reduce to C_{n,Sigma}.
assert np.isclose(
    conventional_spa_logdensity_at_a(np.zeros(d)), C_n_Sigma, atol=2.0e-12
)


# =============================================================================
# 6. WHITENED 2D REDUCTION FOR MODES AND SHELL PROFILES: (8.26), APPENDIX C
# =============================================================================

# Write b=Sigma^{-1/2}a.  By (8.8), (8.9), and Appendix C.4, the whitened law
# is invariant under rotations fixing e.  Consequently every density used in
# Section 8.3 depends only on
#
#     b = z e + y,  y perpendicular to e,  r=||y||.
#
# For evaluation we identify the plane span{e,f_axis} with R^2 and write
# b=(b1,b2), where b1=z and b2=r>=0.

# Appendix C.3 gives sigma^2 M C M = I - sigma^2 v v^T.  Hence the covariance
# M C M has one eigenvalue in the e direction and one repeated eigenvalue in
# every transverse direction.
q1 = 1.0 - sigma**2 * vnorm**2
if q1 <= 0.0:
    raise RuntimeError("Unexpected nonpositive whitened covariance eigenvalue")
qc_parallel = sigma**(-2) - vnorm**2
qc_perp = sigma**(-2)
logdet_MCM = np.log(qc_parallel) + (d - 1.0) * np.log(qc_perp)


def whitened_psi_grad_hess(eta: np.ndarray):
    """Whitened cgf psi, gradient, 2D Hessian, and tilted cbar.

    Here psi(eta)=phi(Sigma^{-1/2} eta), restricted to span{e,f_axis}.
    By whitening, Hess psi(0)=I.
    """
    eta1, eta2 = float(eta[0]), float(eta[1])
    x = vnorm * eta1
    radius2 = eta1 * eta1 + eta2 * eta2

    exponents = np.array(
        [
            sigma * s_minus * x
            + 0.5 * c_minus * (radius2 - sigma**2 * x * x),
            sigma * s_plus * x
            + 0.5 * c_plus * (radius2 - sigma**2 * x * x),
        ]
    )
    logs = np.array([np.log(p_low), np.log(p_high)]) + exponents
    logZ = float(logsumexp(logs))
    weights = np.exp(logs - logZ)

    state_gradients = np.array(
        [
            [c_minus * q1 * eta1 + sigma * s_minus * vnorm, c_minus * eta2],
            [c_plus * q1 * eta1 + sigma * s_plus * vnorm, c_plus * eta2],
        ]
    )
    grad = weights @ state_gradients

    c_weighted = float(weights[0] * c_minus + weights[1] * c_plus)
    gradient_difference = state_gradients[1] - state_gradients[0]
    hess2 = (
        c_weighted * np.diag([q1, 1.0])
        + weights[0]
        * weights[1]
        * np.outer(gradient_difference, gradient_difference)
    )
    return logZ, grad, hess2, c_weighted


def whitened_saddle(b1: float, b2: float, start: Optional[np.ndarray] = None) -> np.ndarray:
    """Solve grad psi(eta)=b in the symmetry-reduced two-dimensional plane."""
    b = np.array([b1, b2], dtype=float)
    if start is None:
        start = b.copy()  # Hess psi(0)=I, so eta=b is a natural central start.
    sol = root(
        lambda eta: whitened_psi_grad_hess(eta)[1] - b,
        start,
        jac=lambda eta: whitened_psi_grad_hess(eta)[2],
        method="hybr",
        options={"xtol": 1.0e-12},
    )
    residual = np.linalg.norm(whitened_psi_grad_hess(sol.x)[1] - b)
    # scipy.root can occasionally report "no progress" at an already converged
    # point (notably b=0).  The residual is the mathematically relevant test.
    if residual > 1.0e-9:
        raise RuntimeError(("Whitened saddle solve failed", sol.message, b, sol.x, residual))
    return sol.x


def exact_logdensity_whitened(b1: float, b2: float) -> float:
    """Exact log pdf of b=Sigma^{-1/2} Xbar_n in the reduced plane."""
    distance = (b1 - coeff_k * vnorm) ** 2 / qc_parallel + b2**2 / qc_perp
    terms = (
        log_pi_k
        - 0.5 * d * np.log(2.0 * np.pi)
        - 0.5 * d * np.log(v_k)
        - 0.5 * logdet_MCM
        - distance / (2.0 * v_k)
    )
    return float(logsumexp(terms))


def conventional_logdensity_whitened(b1: float, b2: float) -> float:
    """Conventional SPA log pdf in whitened coordinates."""
    eta = whitened_saddle(b1, b2)
    psi, _, hess2, c_weighted = whitened_psi_grad_hess(eta)
    sign_h, logdet_h2 = np.linalg.slogdet(hess2)
    if sign_h <= 0:
        raise RuntimeError("Whitened saddle Hessian is not positive definite")

    # The remaining d-2 transverse directions all have eigenvalue c_weighted.
    logdet_H_whitened = logdet_h2 + (d - 2.0) * np.log(c_weighted)
    phi_star = float(eta @ np.array([b1, b2]) - psi)
    return float(
        0.5 * d * np.log(n)
        - 0.5 * d * np.log(2.0 * np.pi)
        - 0.5 * logdet_H_whitened
        - n * phi_star
    )


def b_to_a(b1: float, b2: float) -> np.ndarray:
    """Map reduced whitened coordinates b1*e+b2*f back to a-space."""
    b_vec = b1 * e_axis + b2 * f_axis
    return Sigma_sqrt @ b_vec


# A density transforms under b=Sigma^{-1/2}a by the Jacobian det(Sigma)^{1/2}.
# We add 0.5 log det Sigma to the original-a log densities so that ALL four
# quantities below are densities with respect to db.  Signed log-density errors
# are then obtained by direct subtraction.
WHITENING_LOG_JACOBIAN = 0.5 * logdet_Sigma


def gaussian_logdensity_whitened(b1: float, b2: float) -> float:
    a = b_to_a(b1, b2)
    return gaussian_logdensity_at_a(a) + WHITENING_LOG_JACOBIAN


def cubic_logdensity_whitened(b1: float, b2: float) -> float:
    a = b_to_a(b1, b2)
    return cubic_logdensity_at_a(a) + WHITENING_LOG_JACOBIAN


# Cross-check the 2D conventional formula against the general scalar-saddle
# implementation at a generic point.
_test_b1, _test_b2 = 0.013, 0.017
_test_a = b_to_a(_test_b1, _test_b2)
_test_general = conventional_spa_logdensity_at_a(_test_a) + WHITENING_LOG_JACOBIAN
_test_reduced = conventional_logdensity_whitened(_test_b1, _test_b2)
assert np.isclose(_test_general, _test_reduced, atol=2.0e-10)


# =============================================================================
# 7. MODE LOCATIONS AND TABLE 1: (8.26)--(8.28)
# =============================================================================

# First verify numerically the statement preceding (8.28): the maximizing
# transverse radius is r=0 for all four densities.  We perform a genuine 2D
# constrained optimization in (z,r), r>=0, rather than assuming the result.
MODE_SEARCH_SCALE = 2.0 / np.sqrt(n)

mode_functions = {
    "true": exact_logdensity_whitened,
    "S": conventional_logdensity_whitened,
    "C": cubic_logdensity_whitened,
    "G": gaussian_logdensity_whitened,
}


def verify_transverse_mode_zero(name: str, fun) -> Tuple[float, float]:
    starts = [
        np.array([0.0, 0.0]),
        np.array([-0.25 / np.sqrt(n), 0.25 / np.sqrt(n)]),
        np.array([0.25 / np.sqrt(n), 0.50 / np.sqrt(n)]),
    ]
    best = None
    for start in starts:
        res = minimize(
            lambda zr: -fun(float(zr[0]), float(zr[1])),
            start,
            method="L-BFGS-B",
            bounds=[(-MODE_SEARCH_SCALE, MODE_SEARCH_SCALE), (0.0, MODE_SEARCH_SCALE)],
            options={"maxiter": 300, "ftol": 1.0e-15, "gtol": 1.0e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not best.success:
        raise RuntimeError(f"2D mode check failed for {name}")
    return float(best.x[0]), float(best.x[1])


mode_2d_checks = {
    name: verify_transverse_mode_zero(name, fun) for name, fun in mode_functions.items()
}

# The numerical transverse radii are zero up to optimization tolerance.
for name, (_, r_star) in mode_2d_checks.items():
    if r_star > 1.0e-7:
        raise RuntimeError(f"Unexpected nonzero transverse mode for {name}: r={r_star}")

# Having verified r*=0, polish the axial coordinate by a one-dimensional bounded
# optimization.  This gives the values reported in Table 1 with high accuracy.
mode_z = {}
for name, fun in mode_functions.items():
    if name == "G":
        mode_z[name] = 0.0  # Gaussian mode is exactly at the origin.
    else:
        res = minimize_scalar(
            lambda z: -fun(float(z), 0.0),
            bounds=(-MODE_SEARCH_SCALE, MODE_SEARCH_SCALE),
            method="bounded",
            options={"xatol": 1.0e-14},
        )
        mode_z[name] = float(res.x)

z_true = mode_z["true"]
mode_table = []
for name, display in [
    ("true", "Exact density"),
    ("S", "Conventional SPA"),
    ("C", "Cubic SPA"),
    ("G", "Gaussian"),
]:
    zeta = np.sqrt(n) * mode_z[name]
    delta = np.sqrt(n) * abs(mode_z[name] - z_true)
    mode_table.append(
        {
            "method": display,
            "z_star": mode_z[name],
            "zeta_M": float(zeta),
            "delta_M": float(delta),
            "verified_transverse_r_star": mode_2d_checks[name][1],
        }
    )


# =============================================================================
# 8. c_{3,0}, c_{4,0}: DEFINITION (2.8), APPENDIX C, EQUATION (8.29)
# =============================================================================


def compute_c3_c4_at_center() -> Tuple[float, float, dict]:
    """Compute c_{3,0}, c_{4,0} by Appendix C's four-scalar reduction.

    Definition (2.8) at tau=0 asks for operator norms of the third directional
    derivative of Im phi(i M t) and the fourth directional derivative of
    Re phi(i M t), with ||t||<R sqrt(d/n), M=Sigma^{-1/2}.

    Appendix C.4--C.6 shows that these derivatives depend only on four scalar
    geometric variables.  Parameterize them by

      r     = ||t||,
      a     = angle(v,t),
      b     = angle(v,Theta),  ||Theta||=1,
      gamma = azimuthal angle between the perpendicular parts of t and Theta.

    Then

      x=<v,t>       = ||v|| r cos(a),
      u=<v,Theta>   = ||v|| cos(b),
      w=<t,Theta>   = r[cos(a)cos(b)+sin(a)sin(b)cos(gamma)].

    For fixed (r,a,b,gamma), equation (C.6) gives a scalar function of h,

        phi(iM(t+h Theta)) = log Z(h),

    where each of the two component exponents is a quadratic polynomial in h.
    Derivatives through order four are therefore available analytically.
    """
    radius_max = R_LOCAL * np.sqrt(d / n)

    def log_mixture_derivatives(z):
        r, a_angle, b_angle, gamma = z
        x = vnorm * r * np.cos(a_angle)
        u = vnorm * np.cos(b_angle)
        w = r * (
            np.cos(a_angle) * np.cos(b_angle)
            + np.sin(a_angle) * np.sin(b_angle) * np.cos(gamma)
        )

        # Appendix C.4: <Mt,C Mt>=sigma^{-2}||t||^2-<v,t>^2.
        y0 = sigma**(-2) * r * r - x * x
        y1 = 2.0 * sigma**(-2) * w - 2.0 * x * u
        y2 = sigma**(-2) - u * u

        z_derivs = np.zeros(5, dtype=np.complex128)
        for prob, s_state, c_state in [
            (p_low, s_minus, c_minus),
            (p_high, s_plus, c_plus),
        ]:
            A0 = 1j * sigma * s_state * x - 0.5 * sigma**2 * c_state * y0
            B1 = 1j * sigma * s_state * u - 0.5 * sigma**2 * c_state * y1
            C2 = -0.5 * sigma**2 * c_state * y2
            g0 = prob * np.exp(A0)
            z_derivs[0] += g0
            z_derivs[1] += B1 * g0
            z_derivs[2] += (B1**2 + 2.0 * C2) * g0
            z_derivs[3] += (B1**3 + 6.0 * B1 * C2) * g0
            z_derivs[4] += (B1**4 + 12.0 * B1**2 * C2 + 12.0 * C2**2) * g0

        z0, z1, z2, z3, z4 = z_derivs
        f3 = z3 / z0 - 3.0 * z2 * z1 / z0**2 + 2.0 * (z1 / z0) ** 3
        f4 = (
            z4 / z0
            - 4.0 * z3 * z1 / z0**2
            - 3.0 * (z2 / z0) ** 2
            + 12.0 * z2 * z1**2 / z0**3
            - 6.0 * (z1 / z0) ** 4
        )
        return f3, f4

    def objective3(z):
        f3, _ = log_mixture_derivatives(z)
        return -abs(float(np.imag(f3)))

    def objective4(z):
        _, f4 = log_mixture_derivatives(z)
        return -abs(float(np.real(f4)))

    bounds = [
        (0.0, radius_max),
        (0.0, np.pi),
        (0.0, np.pi),
        (0.0, np.pi),
    ]

    results = []
    for objective in (objective3, objective4):
        res = differential_evolution(
            objective,
            bounds,
            seed=SEED_C3C4,
            popsize=20,
            maxiter=500,
            tol=1.0e-10,
            polish=True,
            workers=1,
            updating="immediate",
        )
        results.append(res)

    c3 = float(-results[0].fun)
    c4 = float(-results[1].fun)
    diagnostics = {
        "c3_argmax_[r,a,b,gamma]": results[0].x.tolist(),
        "c4_argmax_[r,a,b,gamma]": results[1].x.tolist(),
        "c3_function_evaluations": int(results[0].nfev),
        "c4_function_evaluations": int(results[1].nfev),
        "derivatives": "analytic log-mixture derivatives",
    }
    return c3, c4, diagnostics


c3_0, c4_0, c3c4_diagnostics = compute_c3_c4_at_center()


# =============================================================================
# 9. THEOREM 3.1 BOUND AND EXACT CENTRAL SPA ERROR: (8.30)--(8.31)
# =============================================================================

# Insert c_{3,0},c_{4,0} into the explicit algebraic part of the proof of
# Theorem 3.1 at a=0.  As in the paper, exponentially small tail terms are not
# evaluated here.
lambda_star = 1.0 - (c4_0 * R_LOCAL**2 / 2.0) * (d / n)
A0 = c4_0 * d * (d + 2.0) / (24.0 * n)
B0 = c4_0**2 * d * (d + 4.0) * (d + 9.0) / (
    72.0 * lambda_star**4 * n**2
)
D0 = A0 + B0
central_theorem_bound = math.exp(D0) * (
    D0 + (c3_0**2 / (8.0 * lambda_star**3)) * d * (d + 4.0) / n
)

# At a=0, the conventional SPA density is exp(C_{n,Sigma}).  By (2.22),
# exact/conventional = I(0).
exact_logpdf_zero = exact_logdensity_at_a(np.zeros(d))
I0_exact = float(np.exp(exact_logpdf_zero - C_n_Sigma))
central_exact_signed_error = I0_exact - 1.0
central_exact_abs_error = abs(central_exact_signed_error)
bound_over_exact = central_theorem_bound / central_exact_abs_error


# =============================================================================
# 10. MONTE CARLO REFINEMENT: (4.2), (4.21), (8.32)--(8.33)
# =============================================================================


def monte_carlo_J0(
    M_samples: int = MC_SAMPLES,
    seed: int = SEED_MC,
    chunk: int = MC_CHUNK,
) -> Tuple[float, float, int]:
    """Estimate J(0) with the Gaussian Monte Carlo formula (4.21).

    Appendix C.4 implies that at the central saddle the integrand depends on
    S~N(0,I_d) only through

        z = component of S along e,
        ||S||^2.

    Thus it is equivalent, and much cheaper in memory, to draw

        z ~ N(0,1),
        ||S_perp||^2 ~ chi-square_{d-1}

    independently instead of storing M full d-vectors.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    total2 = 0.0
    count = 0
    outside_count = 0

    for start in range(0, M_samples, chunk):
        q = min(chunk, M_samples - start)
        z = rng.normal(size=q)
        perp_sq = rng.chisquare(d - 1, size=q)
        r2 = z * z + perp_sq

        # In (4.2), t=S/sqrt(n).  Appendix C.4 uses x=<v,t>.
        x_t = vnorm * z / np.sqrt(n)
        r2_t = r2 / n
        y = sigma**(-2) * r2_t - x_t * x_t

        low = 1j * sigma * s_minus * x_t - 0.5 * sigma**2 * c_minus * y
        high = 1j * sigma * s_plus * x_t - 0.5 * sigma**2 * c_plus * y
        phi_it = np.log(p_low * np.exp(low) + p_high * np.exp(high))

        # At a=0, G_0(t)=-phi(iMt).  Equation (4.2) gives
        # q_0(S)=n Re G_0(S/sqrt(n))-||S||^2/2 and
        # theta_0(S)=n Im G_0(S/sqrt(n)).
        q0 = -n * np.real(phi_it) - 0.5 * r2
        theta0 = -n * np.imag(phi_it)
        values = np.exp(-q0) * np.cos(theta0)

        # Characteristic function of U_tilde in (4.21).
        inside = r2 < (R_LOCAL**2 * d)
        outside_count += int(np.count_nonzero(~inside))
        values *= inside

        total += values.sum(dtype=np.float64)
        total2 += np.square(values).sum(dtype=np.float64)
        count += q

    mean = total / count
    sample_variance = (total2 - count * mean * mean) / (count - 1)
    standard_error = float(np.sqrt(sample_variance / count))
    return float(mean), standard_error, outside_count


Jhat, Jhat_standard_error, Jhat_outside = monte_carlo_J0()
Jhat_minus_one = Jhat - 1.0
I0_minus_Jhat_abs = abs(I0_exact - Jhat)


# =============================================================================
# 11. SHELL PROFILES AND FIGURES 2--3: (8.34)--(8.37)
# =============================================================================

# Equation (8.34): on S_r, ||b||=r sqrt(d/n), b=Sigma^{-1/2}a.
# Equation (8.37): xi=<b,e>/||b||.  In our reduced coordinates,
#
#     b1 = rho xi,
#     b2 = rho sqrt(1-xi^2),
#     rho = r sqrt(d/n).
#
# Equation (8.35): Delta_M = log rho_n^M - log rho_n.  The exact curve is zero.
# Equation (8.36) gives Delta_S=-log I(a).


def compute_shell_profile(shell_r: float) -> Dict[str, np.ndarray]:
    rho = shell_r * np.sqrt(d / n)
    delta_S = np.empty_like(XI_GRID)
    delta_C = np.empty_like(XI_GRID)
    delta_G = np.empty_like(XI_GRID)

    for j, xi in enumerate(XI_GRID):
        b1 = rho * xi
        b2 = rho * np.sqrt(max(0.0, 1.0 - xi * xi))
        exact = exact_logdensity_whitened(b1, b2)
        delta_S[j] = conventional_logdensity_whitened(b1, b2) - exact
        delta_C[j] = cubic_logdensity_whitened(b1, b2) - exact
        delta_G[j] = gaussian_logdensity_whitened(b1, b2) - exact

    return {
        "xi": XI_GRID.copy(),
        "exact": np.zeros_like(XI_GRID),
        "conventional": delta_S,
        "cubic": delta_C,
        "gaussian": delta_G,
    }


shell_profiles = {r: compute_shell_profile(r) for r in SHELL_RADII}


def shell_title(r: float) -> str:
    if np.isclose(r, 1.0 / 6.0):
        return r"Shell $r=1/6$"
    if np.isclose(r, 1.0 / 3.0):
        return r"Shell $r=1/3$"
    if np.isclose(r, 1.0):
        return r"Shell $r=1$"
    if np.isclose(r, 4.0 / 3.0):
        return r"Shell $r=4/3$"
    return rf"Shell $r={r:g}$"


def save_shell_profile(shell_r: float, basename: str):
    """Save ONE shell profile as one manuscript source image.

    The LaTeX manuscript places two separate source images in Figure 2 and two
    separate source images in Figure 3.  We deliberately preserve that file
    structure instead of combining the profiles into multi-panel graphics.
    """
    profile = shell_profiles[shell_r]
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    ax.plot(profile["xi"], profile["exact"], label="Exact density")
    ax.plot(profile["xi"], profile["conventional"], label="Conventional SPA")
    ax.plot(profile["xi"], profile["cubic"], label="Cubic SPA")
    ax.plot(profile["xi"], profile["gaussian"], label="Gaussian")
    ax.set_title(shell_title(shell_r))
    ax.set_ylabel(r"log-density error $\Delta_M$")
    # CURRENT notation from equation (8.37): xi, not the old s.
    ax.set_xlabel(r"angular coordinate $\xi$")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# These filenames are the ones used by the manuscript's \includegraphics calls.
# Figure 2 uses the first two files; Figure 3 uses the last two.
save_shell_profile(1.0 / 6.0, "log_errors_r_1_6")
save_shell_profile(1.0 / 3.0, "log_errors_r_1_3")
save_shell_profile(1.0, "log_errors_r_1")
save_shell_profile(4.0 / 3.0, "log_errors_r_4_3")

# Save the numerical curves used in Figures 2--3.
for shell_r, profile in shell_profiles.items():
    tag = (
        "1_6" if np.isclose(shell_r, 1 / 6)
        else "1_3" if np.isclose(shell_r, 1 / 3)
        else "1" if np.isclose(shell_r, 1)
        else "4_3"
    )
    with (DATA / f"shell_{tag}_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["xi", "Delta_true", "Delta_S", "Delta_C", "Delta_G"])
        for j in range(len(XI_GRID)):
            writer.writerow(
                [
                    profile["xi"][j],
                    profile["exact"][j],
                    profile["conventional"][j],
                    profile["cubic"][j],
                    profile["gaussian"][j],
                ]
            )


# =============================================================================
# 12. FIXED DATA REALIZATION, FOUR RECONSTRUCTIONS, AND FIGURE 4: (8.39)--(8.41)
# =============================================================================

# Draw all n exposures explicitly from (8.1), (8.5), rather than sampling the
# conditional average.  This makes the simulation a literal implementation of
# the data-generating model.
rng_data = np.random.default_rng(SEED_DATA)
noise_sum = np.zeros(d)
realized_high_count = 0
for _ in range(n):
    is_high = rng_data.random() < p_high
    realized_high_count += int(is_high)
    S_state = s_plus if is_high else s_minus
    c_state = c_plus if is_high else c_minus
    Z = rng_data.normal(size=d)
    Xj = sigma * (S_state * mu + np.sqrt(c_state) * (G_alpha @ Z))  # (8.5)
    noise_sum += Xj

ybar = m_true + noise_sum / n


# Exact likelihood (8.14) and analytic score with respect to theta.
def exact_loglik_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    a = ybar - B @ theta
    Cinv_a = C_alpha_inv @ a
    a_Cinv_a = a @ Cinv_a
    mu_Cinv_a = mu @ Cinv_a
    distance_sq = a_Cinv_a - 2.0 * coeff_k * mu_Cinv_a + mean_quadratic_k
    component_logs = log_pi_k + log_v_normalizer - distance_sq / (2.0 * v_k)
    log_mixture = logsumexp(component_logs)
    value = float(TRUE_CONSTANT + log_mixture)

    posterior_k = np.exp(component_logs - log_mixture)
    posterior_over_v = posterior_k / v_k
    grad_a = (
        -Cinv_a * np.sum(posterior_over_v)
        + Cinv_mu * np.sum(coeff_k * posterior_over_v)
    )
    grad_theta = -B.T @ grad_a
    return value, grad_theta


def gaussian_loglik_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    a = ybar - B @ theta
    u = Sigma_inv @ a
    value = C_n_Sigma - (n / 2.0) * (a @ u)
    grad_theta = B.T @ (n * u)
    return float(value), grad_theta


def cubic_loglik_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    a = ybar - B @ theta
    u = Sigma_inv @ a
    value = cubic_logdensity_at_a(a)

    # Appendix B: T3[u,u,.].
    z = mu @ u
    C_u = C_alpha @ u
    u_C_u = u @ C_u
    T_uu_dot = sigma**3 * (
        kappa3 * z**2 * mu
        + beta * (u_C_u * mu + 2.0 * z * C_u)
    )
    grad_a = -n * u + (n / 2.0) * (Sigma_inv @ T_uu_dot) - 0.5 * (Sigma_inv @ q_vec)
    grad_theta = -B.T @ grad_a
    return float(value), grad_theta


def conventional_loglik(theta: np.ndarray) -> float:
    """Conventional-SPA likelihood (8.16) evaluated at a_theta=ybar-B theta."""
    return conventional_spa_logdensity_at_a(ybar - B @ theta)


def maximize_with_gradient(fun, start: np.ndarray, label: str):
    """Maximize a function returning (value,gradient) with L-BFGS-B."""
    result = minimize(
        lambda th: -fun(th)[0],
        np.asarray(start),
        jac=lambda th: -fun(th)[1],
        method="L-BFGS-B",
        options=OPT_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(f"{label} optimization failed: {result.message}")
    return result


# B has orthonormal columns, so B^T ybar is a convenient common start.
start_theta = B.T @ ybar
res_G = maximize_with_gradient(gaussian_loglik_and_grad, start_theta, "Gaussian")
res_C = maximize_with_gradient(cubic_loglik_and_grad, res_G.x, "Cubic SPA")
res_true = maximize_with_gradient(exact_loglik_and_grad, res_C.x, "Exact likelihood")

# The conventional SPA has a nested scalar saddlepoint solve.  SciPy's
# L-BFGS-B finite-difference gradient is inexpensive here (16 parameters and a
# one-dimensional internal root) and avoids coding a third-cumulant expression
# for the derivative of log det H(tau).  Starting at the exact MLE makes this
# optimization particularly short.
res_S = minimize(
    lambda th: -conventional_loglik(th),
    res_true.x,
    method="L-BFGS-B",
    options={"maxiter": 250, "ftol": 1.0e-13, "gtol": 1.0e-8, "maxls": 40},
)
if not res_S.success:
    raise RuntimeError(f"Conventional SPA optimization failed: {res_S.message}")

# Equation (8.41): definitions of the four maximizers and reconstructed images.
theta_hat = {
    "true": res_true.x,
    "S": res_S.x,
    "C": res_C.x,
    "G": res_G.x,
}
m_hat = {name: B @ theta for name, theta in theta_hat.items()}

# Equation (8.40), in the order M=true,S,C,G used in the paper.
noisy_relative_L2 = float(np.linalg.norm(ybar - m_true) / np.linalg.norm(m_true))
relative_L2 = {
    name: float(np.linalg.norm(m_hat[name] - m_true) / np.linalg.norm(m_true))
    for name in ["true", "S", "C", "G"]
}

# Figure 4 displays only true, averaged noisy, exact-likelihood reconstruction,
# and conventional-SPA reconstruction.  Cubic and Gaussian are omitted in the
# paper because they are visually indistinguishable at this scale, but their
# errors are still computed above and reported in the summary.
images = [m_true, ybar, m_hat["true"], m_hat["S"]]
image_titles = ["True", "Averaged noisy", "Exact likelihood", "Conventional SPA"]
vmin = min(arr.min() for arr in images)
vmax = max(arr.max() for arr in images)
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8))
for ax, arr, title in zip(axes, images, image_titles):
    h = ax.imshow(arr.reshape(m, m), vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(h, ax=axes.tolist(), fraction=0.022, pad=0.02)
# Keep the manuscript/source filename unchanged.
fig.savefig(FIG / "reconstruction_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 13. IMPLEMENTATION / CONSISTENCY CHECKS
# =============================================================================


def finite_difference_gradient(value_fun, theta: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    """Centered finite-difference gradient used only as a coding check."""
    g = np.empty_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = step
        g[j] = (value_fun(theta + e) - value_fun(theta - e)) / (2.0 * step)
    return g


check_theta = res_true.x + 0.001 * np.arange(1, p + 1) / p

def exact_value(th):
    return exact_loglik_and_grad(th)[0]


def cubic_value(th):
    return cubic_loglik_and_grad(th)[0]


def gaussian_value(th):
    return gaussian_loglik_and_grad(th)[0]


gradient_check = {}
for name, pair_fun, value_fun in [
    ("true", exact_loglik_and_grad, exact_value),
    ("C", cubic_loglik_and_grad, cubic_value),
    ("G", gaussian_loglik_and_grad, gaussian_value),
]:
    analytic = pair_fun(check_theta)[1]
    numerical = finite_difference_gradient(value_fun, check_theta)
    gradient_check[name] = float(np.max(np.abs(analytic - numerical)))


# =============================================================================
# 14. SAVE TABLES, NUMERICAL DATA, AND A HUMAN-READABLE SUMMARY
# =============================================================================


def write_records_csv(path: Path, rows: List[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_records_csv(DATA / "table1_modes.csv", mode_table)

# Reconstruction errors in exactly the order displayed in (8.40).
reconstruction_rows = [
    {"quantity": "averaged noisy", "relative_L2_error": noisy_relative_L2},
    {"quantity": "true", "relative_L2_error": relative_L2["true"]},
    {"quantity": "S", "relative_L2_error": relative_L2["S"]},
    {"quantity": "C", "relative_L2_error": relative_L2["C"]},
    {"quantity": "G", "relative_L2_error": relative_L2["G"]},
]
write_records_csv(DATA / "reconstruction_errors.csv", reconstruction_rows)

summary_json = {
    "paper_version": "High-d CLT - v3 (20260930-215525)",
    "equation_8_23_parameters": {
        "m": m,
        "d": d,
        "n": n,
        "alpha": alpha,
        "sigma": sigma,
        "c_minus": c_minus,
        "c_plus": c_plus,
        "d_squared_over_n": d * d / n,
    },
    "figure_1_mu": {
        "seed": SEED_MU,
        "norm_mu": float(np.linalg.norm(mu)),
    },
    "table_1_modes": mode_table,
    "equation_8_29_c3_c4": {"c3_0": c3_0, "c4_0": c4_0},
    "equation_8_30_bound_tail_omitted": central_theorem_bound,
    "equation_8_31_exact_central_error": {
        "I0": I0_exact,
        "I0_minus_1": central_exact_signed_error,
        "abs_I0_minus_1": central_exact_abs_error,
        "bound_over_exact": bound_over_exact,
    },
    "equations_8_32_8_33_monte_carlo": {
        "seed": SEED_MC,
        "M": MC_SAMPLES,
        "chunk": MC_CHUNK,
        "Jhat": Jhat,
        "Jhat_minus_1": Jhat_minus_one,
        "ordinary_MC_standard_error": Jhat_standard_error,
        "outside_U_tilde_count": Jhat_outside,
        "abs_I0_minus_Jhat": I0_minus_Jhat_abs,
    },
    "equation_8_39_true_image": {
        "max_abs": float(np.max(np.abs(m_true))),
        "target_1_5_over_sqrt_n": float(1.5 / np.sqrt(n)),
    },
    "equation_8_40_relative_L2_errors": {
        "averaged_noisy": noisy_relative_L2,
        **relative_L2,
    },
    "reconstruction_optimization_iterations": {
        "true": int(res_true.nit),
        "S": int(res_S.nit),
        "C": int(res_C.nit),
        "G": int(res_G.nit),
    },
    "realized_high_state_count": realized_high_count,
    "appendix_C_identity_residual": appendix_C_identity_residual,
    "c3_c4_optimizer_diagnostics": c3c4_diagnostics,
    "gradient_check_max_abs_error": gradient_check,
    "frozen_random_seeds": {
        "mu": SEED_MU,
        "data": SEED_DATA,
        "monte_carlo": SEED_MC,
        "c3_c4_optimizer": SEED_C3C4,
    },
}
(DATA / "paper_numbers.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

# LaTeX-ready body for Table 1.
table1_tex_lines = []
for row in mode_table:
    if row["method"] == "Exact density":
        delta_text = "$0$"
    else:
        delta_text = f"${row['delta_M']:.2e}$"
    table1_tex_lines.append(
        f"{row['method']} & ${row['zeta_M']:.5f}$ & {delta_text} \\\\" 
    )
(DATA / "table1_rows.tex").write_text("\n".join(table1_tex_lines), encoding="utf-8")

report = f"""SECTION 8 COMPLETE REPRODUCIBILITY REPORT
===========================================
Paper: High-d CLT - v3 (20260930-215525)

Equation (8.23): fixed numerical regime
----------------------------------------
m={m}, d={d}, p={p}, n={n}, d^2/n={d*d/n:.6f}
alpha={alpha}, sigma={sigma}, c_-={c_minus}, c_+={c_plus}
||mu||={np.linalg.norm(mu):.12f}
realized high-state count in Figure 4 data={realized_high_count}/{n}

Figure 1 / equation (8.24)
--------------------------
mu seed={SEED_MU}
||mu||={np.linalg.norm(mu):.12f}

Table 1 / equations (8.26)--(8.28)
-----------------------------------
Exact density       zeta={mode_table[0]['zeta_M']:.8f}   delta={mode_table[0]['delta_M']:.8e}
Conventional SPA    zeta={mode_table[1]['zeta_M']:.8f}   delta={mode_table[1]['delta_M']:.8e}
Cubic SPA           zeta={mode_table[2]['zeta_M']:.8f}   delta={mode_table[2]['delta_M']:.8e}
Gaussian            zeta={mode_table[3]['zeta_M']:.8f}   delta={mode_table[3]['delta_M']:.8e}
Largest numerically verified transverse r_*={max(row['verified_transverse_r_star'] for row in mode_table):.3e}

Equation (8.29)
---------------
c_3,0={c3_0:.10f}
c_4,0={c4_0:.10f}

Equations (8.30)--(8.31)
------------------------
explicit bound, e.s.t. omitted = {central_theorem_bound:.10f}
I(0)-1                         = {central_exact_signed_error:.12e}
|I(0)-1|                       = {central_exact_abs_error:.12e}
bound / exact                  = {bound_over_exact:.6f}

Equations (8.32)--(8.33): Monte Carlo refinement
-------------------------------------------------
M={MC_SAMPLES}, seed={SEED_MC}, chunk={MC_CHUNK}
Jhat_M(0)-1                    = {Jhat_minus_one:.12e}
ordinary MC standard error    = {Jhat_standard_error:.12e}
|I(0)-Jhat_M(0)|               = {I0_minus_Jhat_abs:.12e}
outside U_tilde draws          = {Jhat_outside}

Equation (8.39)
---------------
max_x |m_theta0(x)| = {np.max(np.abs(m_true)):.12e}
1.5/sqrt(n)          = {1.5/np.sqrt(n):.12e}

Equation (8.40): relative L2 errors
------------------------------------
averaged noisy     = {noisy_relative_L2:.10f}
true likelihood    = {relative_L2['true']:.10f}
conventional SPA   = {relative_L2['S']:.10f}
cubic SPA          = {relative_L2['C']:.10f}
Gaussian           = {relative_L2['G']:.10f}

Generated manuscript source images
----------------------------------
Figure 1: {FIG / 'mu.png'}
Figure 2, top:    {FIG / 'log_errors_r_1_6.png'}
Figure 2, bottom: {FIG / 'log_errors_r_1_3.png'}
Figure 3, top:    {FIG / 'log_errors_r_1.png'}
Figure 3, bottom: {FIG / 'log_errors_r_4_3.png'}
Figure 4: {FIG / 'reconstruction_comparison.png'}

NOTE: The four shell profiles remain FOUR separate source-image files, as in
      the manuscript.  All use the x-axis label "angular coordinate xi" with
      the mathematical symbol $\\xi$, as in equation (8.37).

Generated data
--------------
Table 1: {DATA / 'table1_modes.csv'}
Reconstruction errors: {DATA / 'reconstruction_errors.csv'}
Shell profile CSVs: {DATA / 'shell_*_profile.csv'}
Complete JSON summary: {DATA / 'paper_numbers.json'}

Implementation diagnostics
--------------------------
Appendix C identity residual = {appendix_C_identity_residual:.3e}
max gradient-check error     = {max(gradient_check.values()):.3e}
"""

(DATA / "summary.txt").write_text(report, encoding="utf-8")
print(report)
print(f"Total runtime: {time.time() - START_TIME:.2f} seconds")
