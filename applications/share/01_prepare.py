"""
01_prepare.py -- SHARE data for FOCaL: baseline chronic-condition status -> quality-of-life trajectories.

Input: data/raw/data_for_fasten.Rdata (the smoothed SHARE curves of the FASTEN preprocessing:
37 variables x 1518 subjects x 192 months, 2004-2020; subjects with at least seven of eight waves).
Only smoothed curves are available locally, no raw wave-level measurements.

Design (Choice 1 of NOTES.md): the treatment is the chronic-condition status AT BASELINE (month 0),
A = 1 if the subject reports the condition at the first wave, A = 0 otherwise. Controls may develop
the condition later; nothing after baseline enters the definition of the groups. The previous
analysis compared subjects with the condition at every wave against subjects who never developed
it, which uses future disease status (Reviewer 2, Major 6; Reviewer 1); that contrast is kept
only as a secondary analysis ('never_vs_always' masks saved here).

Outputs: data/processed/share.npz with, for hypertension and high cholesterol,
  A_<cond> (baseline status), A_<cond>_always, A_<cond>_never (secondary contrast),
  onset_month_<cond> (first month with the condition, -1 if never),
and the common X (6 baseline covariates), Y_casp, Y_mobility (1518 x 192), months.
Run from the share/ folder:  python3 01_prepare.py
"""

import numpy as np
import pyreadr

r = pyreadr.read_r('data/raw/data_for_fasten.Rdata')
A_mat = np.array(r['Amat'])                                   # (37, 1518, 192)
names = [str(v) for v in np.array(r['names_var']).ravel()]
idx = {v: k for k, v in enumerate(names)}
n, m = A_mat.shape[1], A_mat.shape[2]
months = np.arange(m)

# Baseline covariates (constant over time in the Rdata; checked below)
COVS = ['age', 'eduyears_mod', 'n_children', 'gender', 'vaccinated', 'ever_smoked']
X_cols = ['age', 'education_years', 'n_children', 'female', 'vaccinated_childhood', 'ever_smoked']
X = np.column_stack([A_mat[idx[v]][:, 0] for v in COVS])
for v in COVS:
    assert np.allclose(A_mat[idx[v]], A_mat[idx[v]][:, [0]]), f'{v} varies over time'

out = dict(X=X, X_cols=np.array(X_cols), months=months,
           Y_casp=A_mat[idx['casp']], Y_mobility=A_mat[idx['mobilityind']])
lines = [f'n = {n} subjects, m = {m} months', '']
for cond in ('hypertension', 'high_cholesterol'):
    W = A_mat[idx[cond]].round().astype(int)                  # 0/1 status curve (smoothing created values near 0/1)
    assert set(np.unique(W)) <= {0, 1}
    base = W[:, 0]
    never, always = np.all(W == 0, axis=1), np.all(W == 1, axis=1)
    onset = np.where(W.any(axis=1), W.argmax(axis=1), -1)
    monotone = (np.diff(W, axis=1) >= 0).all(axis=1)
    out[f'A_{cond}'] = base
    out[f'A_{cond}_always'] = always
    out[f'A_{cond}_never'] = never
    out[f'onset_month_{cond}'] = onset
    dev = (base == 0) & W.any(axis=1)
    lines += [f'{cond}:',
              f'  baseline status A = 1: {base.sum()}, A = 0: {(base == 0).sum()}',
              f'  condition at every wave: {always.sum()}; never: {never.sum()}; '
              f'free at baseline with later onset: {dev.sum()} (median onset month {np.median(onset[dev]):.0f})',
              f'  status monotone in time for all subjects: {monotone.all()}', '']
np.savez('data/processed/share.npz', **out)
with open('data/processed/counts.txt', 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('covariate means (all / A=1 / A=0, hypertension):')
A = out['A_hypertension']
for k, c in enumerate(X_cols):
    print(f'  {c:22s} {X[:, k].mean():7.2f} {X[A == 1, k].mean():7.2f} {X[A == 0, k].mean():7.2f}')
