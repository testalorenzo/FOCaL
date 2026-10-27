# FOCaL: Functional Outcome Causal Learning

Code for the paper

> F. Salmaso, L. Testa, F. Chiaromonte. *A Doubly Robust Machine Learning Approach for Disentangling Treatment Effect Heterogeneity with Functional Outcomes.* [arXiv:2602.11118](https://arxiv.org/abs/2602.11118)

**FOCaL** is a doubly robust meta-learner for the *functional conditional average treatment effect* (F-CATE): the effect of a binary treatment on a functional outcome, as a function of covariates. It proceeds in three cross-fitted stages: (1) nuisance estimation (functional outcome regressions and propensity score), (2) doubly robust functional pseudo-outcomes, (3) a final regression of the pseudo-outcome difference on the covariates. When the third stage is a linear smoother, FOCaL also returns global confidence sets for the whole effect curve (L2 balls, Proposition 4.6 of the paper).

## Contents

```
FOCaL.py                 the estimator (class FOCaL) and linear-smoother third stages (FixedBasisLS, ...)
simple_example.py        fit, predict and build a confidence set on simple synthetic data
simulations/             the simulation study of the paper
  simulations.py         base design and one-factor-at-a-time variations (Methods, "Simulation study")
  figure2.py             main-text figure (Figure 2)
  figure_supp.py         supplementary figures (sample size, variations, boosting nuisances, diagnostics)
  table_bands.py         supplementary table comparing L2 sets with grid bands
  results/               per-trial tables (gzipped), summaries and figures of the runs reported in the paper
applications/
  focal_apps.py          shared helpers (nuisance learners, spline third stage, inference, plotting)
  encode/                CpG islands and H3K4me3 profiles around gene starts (ENCODE, GM12878)
  share/                 chronic conditions and quality-of-life trajectories (SHARE)
```

## Installation

Python 3.10 or later. Install the dependencies with

```
pip install -r requirements.txt
```

`pyreadr` is needed only to prepare the SHARE data and `pyBigWig` only to preprocess the raw ENCODE signal; both can be skipped otherwise.

## Using the estimator

```python
from FOCaL import FOCaL, FixedBasisLS
from sklearn.linear_model import LinearRegression, LogisticRegression

focal = FOCaL(nuisance_mu=LinearRegression(),          # outcome regressions, fit per arm on fPCA scores
              nuisance_pi=LogisticRegression(),        # propensity score, clipped to (eps, 1 - eps)
              final_regression=FixedBasisLS(),         # third stage: least squares on (1, x), a linear smoother
              fpca=10, fpca_third_stage=False, eps=0.01)
focal.fit(X, X, A, Y, n_folds=5)                        # Y: (n, m) curves on a common grid
theta_hat = focal.predict(X_new)                        # F-CATE curves at new covariate values
fate_hat = focal.get_FATE()                             # FATE: mean of the pseudo-outcome differences
cs = focal.confidence_set(X_new, alpha=0.05, dt=dt)     # 95% L2 balls: cs['radius'] (dt = quadrature weight of the grid)
fcs = focal.fate_confidence_set(alpha=0.05, dt=dt)
```

* `nuisance_mu` is any scikit-learn regressor; with `fpca=K` it is fit on the first K functional principal component scores of the training curves (with `fpca=0.99`, on as many scores as needed to explain 99% of their variance), otherwise on the grid. `nuisance_pi` is any classifier with `predict_proba`.
* `final_regression` is any regressor; confidence sets require a linear smoother exposing a `weights(X)` method. `FixedBasisLS(basis, ridge)` is least squares on a fixed basis (see `applications/focal_apps.py` for additive cubic B-splines); `KernelRidgeSmoother` and `KNNSmoother` are also provided.
* `set_mode('plugin')` or `set_mode('ipw')` switches to the outcome regression (T-learner) or inverse probability weighting pseudo-outcomes without refitting the nuisances.
* Confidence sets are calibrated by simulating Gaussian curves with the residual covariance of the third stage (Supplementary Eq. for Sigma_hat) and taking the (1 - alpha)-quantile of their L2 norms.

`python simple_example.py` runs a complete example.

## Reproducing the simulation study

From the `simulations/` folder:

```
python simulations.py --n_trials 1000               # base design (about 10 minutes on 8 cores)
python simulations.py --variations --n_trials 200   # 16 design variations (about 1 hour)
python figure2.py                                   # Figure 2 of the paper
python figure_supp.py                               # supplementary figures
python table_bands.py                               # supplementary table on inference procedures
```

`simulations.py --out DIR` writes to another folder (useful for tests, e.g. `--n_trials 2`). The per-trial tables of the runs reported in the paper are stored gzipped in `results/` and read transparently by the figure scripts, so the figures can be regenerated without rerunning the study.

## Reproducing the applications

**ENCODE** (all data public). From `applications/encode/`:

```
bash 01_download.sh        # ~700 MB from ENCODE, GENCODE and UCSC
python 02_preprocess.py    # -> data/processed/encode.npz (included in the repository)
python 03_fit.py           # main fit, overlap diagnostics, targets, seeds, secondary fit
python 06_figure_main_text.py
python 07_supp.py          # basis sensitivity, secondary analysis, supplementary table and figures
```

Steps 1 and 2 can be skipped, since the processed data are included. `NOTES.md` documents every choice and result.

**SHARE**. The analysis starts from the smoothed curves of the FASTEN preprocessing (`data/raw/data_for_fasten.Rdata`, included: 1518 subjects, 37 variables on a grid of 192 months); the raw SHARE records are not redistributed and can be requested at https://share-eric.eu/data. From `applications/share/`:

```
python 01_prepare.py       # baseline treatment, covariates, outcomes -> data/processed/share.npz
python 02_fit.py           # FOCaL for hypertension and high cholesterol, CASP and mobility index
python 03_diagnostics.py   # overlap, balance, sensitivity analyses, unmeasured confounding
python 04_figure.py        # main-text and supplementary figures
python 05_supp_table.py    # sensitivity table
```

The summary tables and figures of the paper are included in `applications/share/results/`; the intermediate `.npz` files are regenerated by the scripts.

## Citation

```
@article{salmaso2026focal,
  title   = {A Doubly Robust Machine Learning Approach for Disentangling Treatment Effect Heterogeneity with Functional Outcomes},
  author  = {Salmaso, Filippo and Testa, Lorenzo and Chiaromonte, Francesca},
  journal = {arXiv preprint arXiv:2602.11118},
  year    = {2026}
}
```
