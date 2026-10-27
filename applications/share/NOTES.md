# SHARE application: chronic condition at baseline and quality-of-life trajectories

Rebuilt 2026-10-25 in response to Reviewer 2 (Major 6) and Reviewer 1 (overstated causal claims,
selection on future disease status). Scripts: `01_prepare.py`, `02_fit.py`, `03_diagnostics.py`,
`04_figure.py` (run from this folder, in this order). The estimator is `../../FOCaL.py` through
`../focal_apps.py`.

## Data

`data/raw/data_for_fasten.Rdata`: the smoothed SHARE curves of the FASTEN preprocessing
(Boschi et al.), 37 variables x 1518 subjects x 192 months (2004-2020), subjects with at least seven
of the eight waves. Only smoothed curves are available locally; the raw wave-level EasySHARE records
are not, so reconstruction error against the raw measurements cannot be computed here and the
seven-wave filter cannot be relaxed (see Limitations).

## Target trial and choices

1. **Treatment = status at baseline.** A = 1 if the subject reports the condition (hypertension; high
   cholesterol in the supplement) at the first interview, A = 0 otherwise. Nothing observed after
   baseline enters the group definition. Controls may develop the condition later (522 of 1099 for
   hypertension, median onset month 72; 458 of 1205 for high cholesterol): the estimand is the effect
   of carrying a diagnosed condition at baseline on the following 16 years, the observational analogue
   of an intention-to-treat contrast. The previous version compared subjects with the condition at
   every wave (419) against subjects who never developed it (577), discarding the 522 later-onset
   subjects: that conditions on future disease status and is kept only as a **secondary analysis**
   (`*_secondary_*` rows in `results/summary.csv`, dashed curve in the figure).
2. **Target trial.** Eligibility: SHARE respondents aged 40-82 at the 2004 wave who took part in at
   least seven waves (the sample available; a survivor and respondent cohort, see Limitations).
   Strategies: "diagnosed hypertension at enrollment" vs "no diagnosed hypertension at enrollment".
   Assignment: observational, assumed exchangeable given the baseline covariates of Choice 3.
   Outcomes: CASP (quality of life, higher = better) and mobility index (higher = more impairment)
   over the 192 months after enrollment. Estimands: FATE and F-CATE along age and gender.
3. **Covariates (baseline, time-constant in the data):** age, years of education, number of children,
   gender, childhood vaccinations, ever smoked. As in the previous version; measured at or before
   baseline. Baseline values of time-varying health variables (BMI, EURO-D, ...) are not used: they
   are plausibly affected by a diagnosis made before enrollment.
4. **Nuisances:** as in the simulations and ENCODE: fPCA compression keeping 99% of the variance of the
   training curves (6 components in every fold for both outcomes: the smoothed curves are very low
   rank), early-stopped gradient boosting (at most 500 trees, rate 0.05, 15 leaves, 20 per leaf,
   10% held out) per score and arm; boosting classifier for the propensity, clipped to (0.02, 0.98);
   5 folds balanced on A, seed 0.
5. **Third stage:** least squares on additive cubic B-splines in the standardized covariates with 6
   knots at the quantiles (binary covariates linear), ridge 0.01 on non-intercept coefficients. Six
   knots is the `focal_apps` default used for the applications of comparable size (NHANES, LCL);
   ENCODE uses 15 with 13 times more units. A first run with 10 knots gave the same FATE (the FATE
   does not depend on the third stage) and surfaces with visible knot-driven ripples along age
   (about 300 subjects per fold); 10 and 15 knots are kept in `results/sensitivity_*.csv`. The basis
   is additive: gender shifts the level of the effect, and the old - young contrast is the same for
   men and women by construction, so no gender-specific age profile is claimed.
6. **Inference:** L2 balls of Proposition 4.5 (alpha = 0.05, norm on [0, 1] with dt = 1/(m-1), 20,000
   Gaussian draws), for the FATE (weights 1/n, residuals around the sample mean), the cuts (age at its
   10/50/90% quantiles, 51/59/69 years, for men and women, other covariates at their medians) and the
   contrasts old - young within gender and women - men at median age (differences of smoother
   weights). Significance = zero not in the set (norm > radius).
7. **Diagnostics reported** (`03_diagnostics.py`): propensity quantiles and clipped share; inverse
   probability weights and effective sample sizes per arm; standardized mean differences before and
   after weighting; cross-fitted outcome predictions against observed means and nuisance residual
   RMSE; third-stage residual RMSE; FATE refits with seeds 1 and 2; fPCA at 90/95/99%.
8. **Re-smoothing sensitivity.** The curves are read at the eight approximate wave months (0, 30, 54,
   84, 108, 132, 156, 186 after the first interview) and re-smoothed with cubic smoothing splines
   under GCV (as in the original preprocessing) and fixed penalties 1e1, 1e3, 1e5; FOCaL is refit on
   each version. This probes the smoothing choice, not the imputation of missing waves.
9. **Unmeasured confounding.** For the FATE: (i) the share of the estimated effect that would have to
   be confounding bias for zero to enter the L2 set, 1 - radius/||FATE||; (ii) the constant shift in
   outcome units that brings zero into the set; (iii) an E-value at the month of the largest
   standardized effect, with RR = exp(0.91 d) (VanderWeele and Ding, 2017).

## Limitations (to be stated in the paper)

- Survivor/respondent cohort (at least seven waves) and smoothing from eight interviews: the
  observation layer is outside the theory (Discussion); the sparse-noisy simulation shows what it
  does to coverage.
- "Diagnosed hypertension at baseline" is a status, not a manipulable intervention; the contrast is
  descriptive of the population with and without a diagnosis at enrollment.
- Controls who develop the condition later dilute the contrast toward zero (compare the secondary
  analysis).

### Nuisance-learner sensitivity (added 2026-10-25, Choice 10; `sensitivity_*.csv`, rows "nuisances")
Main = gradient boosting. Alternatives: OLS on the fPCA scores + logistic regression; random forests
(300 trees, min leaf 10) for both; the previous version's two-layer MLP (10, 10, logistic, alpha 1e-3)
for the outcomes + logistic regression. Hypertension: CASP FATE L2 0.67-0.70 vs radius 0.57-0.58 (all
reject; pointwise deviation from the main curve up to 0.33), mobility 0.14-0.15 vs 0.07 (all reject,
deviation <= 0.04). High cholesterol: CASP rejects under all learners; mobility rejects only with
linear nuisances (RF 0.065 vs 0.073, NN 0.055 vs 0.070 do not) -> "not established" stands.
