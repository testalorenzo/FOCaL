# ENCODE application: CpG island at the promoter and the H3K4me3 profile around the TSS

Effect of a CpG island at the promoter (binary A) on the H3K4me3 fold-change profile
over TSS - 2 kb .. TSS + 2 kb (functional outcome Y(t), t = genomic position) of
protein-coding genes in GM12878. Main analysis (decided 2026-10-20): pre-treatment covariates only (total effect), heterogeneity along the distance to the nearest other gene start; the fit with expression in X (effect among genes of equal activity) is secondary.
Scripts: `01_download.sh`, `02_preprocess.py`, `03_fit.py`, `04_plot.py`, `05_plot_l2.py`, `06_figure_main_text.py` (run from this folder).

## Data

Downloaded 2026-10-08 (all GRCh38 / GENCODE v29).

| file | source | accession |
|---|---|---|
| `data/raw/H3K4me3_GM12878_fc.bigWig` | ENCODE, H3K4me3 ChIP-seq GM12878, fold change over control, pooled replicates | ENCFF927KAJ |
| `data/raw/rnaseq_GM12878_gene_quant.tsv` | ENCODE, RNA-seq GM12878, RSEM gene quantification (GENCODE V29 ids) | ENCFF413MYB |
| `data/raw/gencode.v29.basic.annotation.gtf.gz` | GENCODE v29 basic annotation | release_29 |
| `data/raw/cpgIslandExt_hg38.txt.gz` | UCSC Genome Browser, table `cpgIslandExt`, hg38 | - |

Processed data: `data/processed/encode.npz` (Y 19864 x 80, A, X 19864 x 5, gene ids, TSS, bin positions);
per-gene table in `data/temp/genes_table.csv`. The superseded gene-level-TSS version is kept for reference in `data/temp/` (`02_preprocess_v1.py`, `encode_v1.npz`, `summary_v1.csv`, `full_results_v1.npz`).

## Choices

1. **Gene set.** Protein-coding genes (`gene_type == "protein_coding"`, `feature == "gene"`) on chr1-22 and chrX, one TSS per gene. Protein-coding genes give a homogeneous unit with a well-defined TSS and an expression measurement; chrY and unplaced contigs are dropped.
2. **TSS definition.** The analysis TSS is the transcript start shared by most of the gene's APPRIS-principal transcripts (fallback: the modal start over all basic transcripts of the gene). The gene-level start/end from the GTF is NOT used: gene spans include rare long transcripts, so for 20% of protein-coding genes the gene-level TSS lies more than 500 bp (13%: more than 2 kb; ACTB: 33 kb) from the principal promoter, which mis-centres the window and mis-codes canonical CpG-island promoters (ACTB included) as untreated. 3,948 genes (19.9%) move by more than 500 bp under the corrected definition. The gene-level TSS is kept in the npz as `gene_level_tss`. Two external checks: profiles re-extracted from the bigWig for ten genes match the stored outcomes, and after orientation both strands peak downstream of the TSS (at +525 to +625 bp); simple OLS and logistic-IPW adjustments (with expression among the covariates, as in the fit of that date) gave a FATE of the same shape with peaks of 0.80-0.88, within 0.2 of the corresponding FOCaL fit (0.72).
3. **Window and bins.** TSS - 2000 .. TSS + 2000 bp in 80 bins of 50 bp (bin means, `exact=True`); profiles of minus-strand genes are reversed so the gene body is always to the right. 4 kb covers the whole promoter-proximal H3K4me3 peak; 50 bp is well below the nucleosome-scale features of the signal.
4. **Transformation.** Y = log2(1 + fold change). Fold change is heavily right-skewed (max 53); the log tames outliers in the pseudo-outcomes, and 1 + keeps zeros (no signal) at 0.
5. **Treatment.** A = 1 if any UCSC CpG island overlaps TSS +/- 500 bp. This is the standard operational definition of a "CpG-island promoter"; 66.8% of genes are treated.
6. **Covariates** (revised 2026-10-20). Main fit: log10 gene length, number of transcripts in the basic annotation, log10(1 + distance in bp from the analysis TSS to the nearest gene-level TSS of any OTHER gene in the GTF on the same chromosome), strand. These are fixed by the DNA sequence and the annotation, so they are pre-treatment, as Assumption 4.1 and Lemma 4.2 require. Distance uses log10(1 + d) because 50 genes share a TSS position with another gene (d = 0). Expression, log(1 + TPM), is NOT in the main fit: it is partly a consequence of the island (and of H3K4me3), see Caveats; it enters only the secondary fit (effect among genes of equal activity). Heterogeneity variable: the distance to the nearest other gene start, the only covariate along which the F-CATE varies (Choice 13); gene length and transcript count were examined in the same way and show none. The first version (2026-10-08 to 10-20) had expression in X and as heterogeneity variable; its scripts and results are in `data/temp/before_align_2026-10-20/`.
7. **Nuisance models** (revised 2026-10-20, aligned with the simulation study). Gradient boosting for the propensity and, after an fPCA compression of Y that keeps 99% of the variance of the training curves (30 components in every fold), one booster per score for each outcome regression; boosting as in the simulations (at most 500 trees, learning rate 0.05, 15 leaves, 20 observations per leaf, early stopping on a held-out 10% of the training fold); 5-fold cross-fitting with folds stratified on A. The first version kept 10 components, which explain only 92.5% of the variance (median per-curve truncation error 0.33): the outcome regressions then carry a bias that does not vanish with n, against the consistency required by Proposition B.1(ii). Switching to 99% changed the FATE by at most 0.015.
8. **Third stage** (revised 2026-10-20). Least squares of the pseudo-outcome differences on an additive cubic B-spline basis in the standardized covariates, 15 knots at the distinct quantiles of each covariate (ties merge knots: 30% of genes have TPM = 0), strand entering linearly, ridge 0.01 on the non-intercept coefficients; fit on the 80-point grid, so it is a linear smoother and the bands are available. Same settings as the spline third stage of the simulations, fixed a priori. The first version (6 equispaced knots, ridge 1) is a smoother fit whose smoothing bias (Proposition B.3c) is not negligible: at median expression its upstream and downstream lobes differ from the 15-knot fit by up to 0.4 and its bands are 2-2.5 times narrower. Sensitivity over 6, 10, 15 and 25 quantile knots (assessed on the fit with expression in X, then the main one): the FATE does not depend on the basis; the expression cuts keep their level ordering (q50 above q10 and q90) and the TSS dip at q90, while the position of the larger lobe at q50 alternates between -325 and +675 bp (the two lobes are within 0.15 of each other).
9. **Clipping.** Propensities clipped to (0.02, 0.98) (helper default); no gene reached the bounds.
10. **Overlap handling.** Rule fixed in advance: if more than 5% of genes had clipped propensities, refit on the overlap subsample 0.05 <= pi_hat <= 0.95. The clipped share was 0.000, so no subsample was fit and the target population is all 19,864 genes.
11. **alpha and norm.** 0.05 for all bands and L2 balls (simultaneous over the 80 grid points). The outcome is a step function on 80 bins of equal width, so the L2 norm on [0, 1] is sqrt((1/80) sum_t f(t)^2) (dt = 1/m; the first version used 1/(m-1), which inflates norms and radii by 0.6% and changes no test). On this space all norms are equivalent and the sup-type bands follow from Proposition 4.5 by the continuous mapping theorem; the main-text figure uses the L2 balls only.
12. **Seeds.** Seed 0 for everything reported; seeds 1 and 2 refit the whole pipeline (folds, boosting, multiplier draws) for the FATE as a stability check.
13. **Heterogeneity tests** (new 2026-10-20). Differences between cuts, theta_hat(x_k) - theta_hat(x_l), are tested with the same L2 ball and bands: a difference of linear smoothers is a linear smoother with weights omega_i(x_k) - omega_i(x_l). Along the distance: q10 - q50, q10 - q90, q50 - q90. Along gene length and transcript count: q90 - q10. All four continuous covariates were examined (strand is binary and only shifts the additive fit). Basis sensitivity (2026-10-20, scratch; nuisances fixed, third stage refit with 6/10/15/25 quantile knots and the old 6 uniform knots/ridge 1): close - typical and close - far reject under every basis, significant upstream only, with the close profile at -975 bp between 1.25 and 1.38 against 0.53-0.61 for typical; the small differences (typical - far at most 0.2, length and transcripts at most 0.2-0.4) are rejected under the bases with fewer knots (narrower bands) and not under 15 or 25 knots. They are therefore reported as "small", not as "absent".
14. **Secondary fit** (new 2026-10-20). Expression added to X, all other settings unchanged: FATE among genes of equal activity.

## Diagnostics

Counts at each step (`data/processed/counts.txt`):

| step | n |
|---|---|
| genes in the GTF, all types and contigs | 58,721 |
| protein-coding | 19,940 |
| on chr1-22, chrX | 19,864 |
| with a transcript-based TSS | 19,864 |
| with an RNA-seq quantification (all matched; 0 dropped) | 19,864 |
| 4-kb window inside the chromosome | 19,864 |
| **final n** | **19,864** |

Treated (CpG island at the promoter): 13,266 of 19,864 (66.8%). Treated share by expression tercile: 41.3% (low), 72.9% (mid), 87.7% (high). No bigWig bin was missing. Sanity check of the window placement: among expressed genes (TPM > 1), 74.3% have their own H3K4me3 maximum within 1 kb of the analysis TSS (70.3% with the gene-level TSS).

Overlap report of the main fit (`results/overlap.txt`, verbatim):

```
propensity quantiles 1/5/25/50/75/95/99%: [0.133, 0.42, 0.569, 0.668, 0.815, 0.917, 0.935]
share with pi_hat < 0.05: 0.000, > 0.95: 0.002
share at the clipping bounds (0.02, 0.98): 0.000
treated share: 0.668 (n = 19864)
```

Overlap is good: without expression the covariates predict the island only moderately (1-99% of pi_hat in 0.13-0.94, 0.2% above 0.95, none at the bounds). With expression in X (secondary fit) the upper tail is heavier (99th percentile 0.965, 9.6% above 0.95; highly expressed genes almost all have islands) but still no clipping.

Runtime (14-core Apple machine, 24 GB): main fit 99 s; seed refits and the secondary fit about 110 s each; `03_fit.py` alone 443 s.

## Results

Run of 2026-10-20 (main fit: pre-treatment covariates; settings of Choices 7-8). All numbers in `results/summary.csv`; curves in `results/full_results.npz`; figures `results/figure_encode_main` (main text; copy in `FOCaL/gfx/ENCODE.pdf`), `results/fig_main`, `results/fig_l2ball`, `results/fig_data`, `results/fig_overlap`. Effects on the log2(1 + fold change) scale; positions are bin centres in bp relative to the TSS, gene body to the right. Distance quantiles: q10 = 123 bp, q50 = 7,617 bp, q90 = 45,162 bp.

| target | max abs estimate (position) | L2 norm / radius | constant half-width | max-t / q_stud | zero excluded (studentized) |
|---|---|---|---|---|---|
| FATE (DR, seed 0) | 1.480 (+675) | 1.005 / 0.034 | 0.062 | 62.7 / 2.97 | whole window |
| F-CATE, distance q10 (close) | 1.653 (+675) | 1.223 / 0.226 | 0.396 | 12.5 / 2.91 | whole window |
| F-CATE, distance q50 (typical) | 1.807 (+675) | 1.113 / 0.181 | 0.335 | 13.9 / 2.97 | -1625..1975 |
| F-CATE, distance q90 (far) | 1.619 (+675) | 1.050 / 0.172 | 0.313 | 13.3 / 2.96 | whole window |
| contrast close - typical | 0.794 (-925) | 0.386 / 0.197 | 0.357 | 7.5 / 2.97 | -1975..-575 |
| contrast close - far | 0.700 (-975) | 0.348 / 0.188 | 0.339 | 7.1 / 2.97 | -1975..-525 |
| contrast typical - far | 0.203 (+625) | 0.093 / 0.137 | 0.261 | 2.0 / 3.05 | none |
| contrast gene length q90 - q10 | 0.294 (+25) | 0.136 / 0.171 | 0.308 | 2.9 / 2.94 | none |
| contrast transcripts q90 - q10 | 0.175 (+1725) | 0.085 / 0.124 | 0.232 | 2.4 / 2.98 | none |
| FATE, seed 1 / seed 2 | 1.482 / 1.480 (+675) | 1.008 / 1.005 | 0.061 / 0.062 | 63.6 / 62.7 | whole window |
| FATE, plug-in (T-learner) | 1.503 (+625) | 1.025 / - | - | - | - |
| FATE, secondary fit (expression in X) | 0.722 (-425) | 0.455 / 0.033 | 0.068 | 31.5 / 3.08 | -1725..1975 |

*FATE.* A CpG island raises H3K4me3 over the whole 4-kb window: +1.31 at -375 bp, +0.73 to +0.88 at the TSS (nucleosome-depleted region), +1.48 at +675 bp, 0.18 at the upstream and 0.66 at the downstream edge; 2^1.48 = 2.8, i.e. the island nearly triples the mark at the peak. Plug-in vs DR: max difference 0.033. Seeds: curves within 0.009.

*Heterogeneity.* Downstream of the TSS the F-CATE is the same at every distance (lobes 1.62-1.81 at +675). Upstream it depends on the distance to the nearest other gene start: for genes with a neighbour start about 120 bp away the effect stays high far upstream (1.38 at -975, 0.66 at the upstream edge) against 0.61-0.68 at -975 (0.23 at the edge) for typical and far neighbours. Both contrasts with the close profile reject in L2 and exclude zero from -1975 to about -550 bp only. Typical vs far, and the q90 - q10 contrasts along gene length and transcript count, are small (at most 0.2, 0.29 and 0.18) and not significant under the main basis (see Choice 13 for the sensitivity). Interpretation: neighbours read in the opposite direction, see Agreement with the literature. Who the close neighbours are (scratch check on the GTF, all gene types): among genes whose nearest other gene-level TSS is within 300 bp (n = 3,398), 78% have it on the opposite strand (58% strictly head-to-head, i.e. partner TSS upstream; the rest are opposite-strand partners starting just inside the gene body, whose body also extends to the upstream side); 300-1000 bp (n = 1,918): 88% opposite strand. 26.8% of genes have a neighbour start within 1 kb (more than Trinklein's >10% because non-coding genes count as neighbours); their treated share is 70% vs 66%.

*Secondary fit (equal activity).* With expression in X the FATE halves (0.72 at -425, 0.67 at +675, 0.25-0.33 at the TSS); along expression the effect is an inverted U (largest at median expression; contrasts reject). That analysis was the main one until 2026-10-20; its numbers (cuts at expression q10/q50/q90, contrasts) are in the applications report, and it is reproduced by `data/temp/before_align_2026-10-20/03_fit_withTPM_2026-10-20.py`. The pre-alignment version (fPCA 10, 6 equispaced knots) is in the same folder.

## Previous work and novelty

References verified 2026-10-20 (CrossRef / Europe PMC; DOIs below).
- Closest design: Vavouri and Lehner (Genome Biol 2012, 13:R110, 10.1186/gb-2012-13-11-r110) compare chromatin around the TSS of CpG-island and non-CpG promoters matched on expression (low/mid/high classes, CD4+ T cells), and find marks higher upstream specifically at CpG-island promoters (H3K4me3 "to a lesser extent"). Descriptive, no adjustment beyond matching, no inference on curves; it is the reference for the upstream asymmetry found here.
- Functional analyses of the same kind of data are descriptive. Kim and Lin (PLoS One 2020, 15:e0233630) apply FPCA and clustering to four marks jointly (H3K4me2/3, H3K9ac, H4K20me1) around the TSS in GM12878 (10 kb, 100-bp bins) and find upstream, downstream and symmetric active-promoter shapes. Madrigal and Krajewski (BioData Min 2015, 8:20) apply a Karhunen-Loeve transform to 27 ChIP-seq datasets at the TSS in H1 cells.
- MMDiff (Schweikert et al., BMC Genomics 2013, 14:826) tests peak-shape differences between conditions at the same loci, without covariates.
- Functional regression with inference along the genome exists for DNA methylation: Lee and Morris (Bioinformatics 2016, 32:664-672), Zemplenyi et al. (Ann Appl Stat 2021, 15:1366-1385, published version of arXiv:1912.07359).
- Expression prediction from histone marks runs the other way: Karlic et al. (PNAS 2010, 107:2926-2931), Singh et al. DeepChrome (Bioinformatics 2016, 32:i639-i648); Budden and Crampin (BMC Bioinformatics 2016, 17:446) is mainly a distributed implementation, cite only alongside Karlic.
- Doubly robust estimation in genomics: Du, Zeng, Kennedy, Wasserman and Roeder (JASA 2025, 120:2484-2497), many scalar outcomes. The functional DR estimator FOCaL builds on: Testa et al. (J Causal Inference 2026, 14:20250045), applied to SHARE.
- Novelty, tempered: the chromatin of CpG-island promoters, their asymmetry and their bidirectionality are known descriptively (Vavouri and Lehner 2012; Trinklein et al. 2004; Lin et al. 2007); new here are the covariate-adjusted, position-resolved total effect with simultaneous inference, and its heterogeneity tested along every pre-treatment covariate.

## Agreement with the literature

- Treated share 66.8% vs 72% high-CpG promoters (Saxonov, Berg and Brutlag, PNAS 2006, 103:1412-1417; different definition). Islands are much more frequent at highly expressed genes (41/73/88% across expression terciles), as known since Larsen et al. (Genomics 1992, 13:1095-1107).
- Positive effect over the whole window: unmethylated CpG-rich DNA inserted without a promoter acquires Cfp1 and H3K4me3 without detectable Pol II (Thomson et al., Nature 2010, 464:1082-1086; Wachter et al., eLife 2014, 3:e03397; Mendenhall et al., PLoS Genet 2010, 6:e1001244).
- Larger downstream lobe of the total effect and halving at equal activity: transcription deposits H3K4me3 through Cfp1 and H3K4me3 levels depend on activity (Clouaire et al., Genes Dev 2012, 26:1714-1728); CpG islands make chromatin permissive for transcription (Deaton and Bird, Genes Dev 2011, 25:1010-1022). In GM12878, H3K4me3 rises with engaged Pol II (Core et al., Nat Genet 2014, 46:1311-1320).
- Dip at the TSS: nucleosome-depleted region -200..+50 (Barski et al., Cell 2007, 129:823-837).
- Heterogeneity by distance: more than 10% of human genes are in divergent (head-to-head) pairs with TSSs < 1 kb apart, and 77% of these bidirectional promoters lie in a CpG island vs 38% of other promoters (Trinklein et al., Genome Res 2004, 14:62-66, 10.1101/gr.1982804; also Takai and Jones, Mol Biol Evol 2004, 21:463-467; Yang and Elnitski, BMC Genomics 2008, 9(Suppl 2):S3, islands encompassing both TSSs). H3K4me3 is enriched just downstream of each TSS of a bidirectional pair (Lin et al., Genome Res 2007, 17:818-827, 10.1101/gr.5623407). For a gene whose partner starts ~120 bp away, -1 kb is ~+0.9 kb into the partner's body, where the partner's own H3K4me3 sits and is raised by the shared island: the effect becomes roughly symmetric around the start.
- Do not cite for CpG-island claims: Soares et al. (Mol Cell 2017) is a yeast study; Weber et al. (Nat Genet 2007) reports H3K4me2 at inactive CpG-island promoters, not H3K4me3.

## Caveats

- **The "treatment" is a genomic feature, not an intervention.** A = 1 (CpG island) is fixed by the DNA sequence; the causal reading is "the H3K4me3 profile genes with these covariates would have with vs without a promoter CpG island". Insertion experiments (Thomson 2010, Wachter 2014) show the contrast is meaningful as an intervention.
- **Expression is not a pre-treatment covariate**, hence not in the main fit. In GM12878 it is downstream of the island (Deaton and Bird 2011; Clouaire 2012), partly downstream of the outcome (H3K4me3 regulates Pol II pause-release: Wang et al., Nature 2023, 615:339-348; mostly a consequence of transcription: Howe et al., BioEssays 2017, 39:1600095), and shares unmeasured causes with it (transcription-factor and enhancer activity). A broad or germline expression programme as a common cause of the island and the mark (which would make expression a confounder proxy) is not supported by the verified literature (Weber 2007: strong CpG-island promoters stay unmethylated even when inactive; Saxonov 2006: within high-CpG promoters, breadth of expression does not depend on CpG content). The main fit therefore targets the total effect; it rests on no unmeasured confounding given gene length, transcript count, distance to the nearest gene start and strand (gene function, conservation and transcription-factor binding are not measured; transcript count may be mildly affected by the island through alternative promoters). The secondary fit with expression is an effect at a given level of a covariate affected by the treatment (Rosenbaum, JRSS-A 1984, 147:656-666), a controlled-direct-type contrast whose causal reading needs further assumptions.
- **Overlap.** Not restricted: no propensities were clipped; 0.2% exceed 0.95 in the main fit (9.6% in the secondary fit with expression).
- **TSS definition.** One annotated promoter per gene (the principal-transcript TSS); alternative promoters are not modelled, and about a quarter of expressed genes have their H3K4me3 maximum more than 1 kb from the analysis TSS. The earlier gene-level definition changed the qualitative heterogeneity pattern, so this is the most consequential preprocessing decision.
- **Signal and units.** Fold change over control from pooled replicates on a log2(1 + x) scale; specific to GM12878 and to the ENCODE H3K4me3 antibody and pipeline. The raw H3K4me3 maximum sits at +525..+625 bp, further downstream than the +50/+210/+360 nucleosome peaks of Barski et al. (2007); plausibly fold-change and fragment-extension processing (not verified).
- **Bands.** Simultaneous over the 80-point grid, conditional on the spline basis and the nuisance tuning. The basis was fixed to the simulation settings, not chosen from these responses, but the knot sensitivity in Choice 8 was looked at; the cut bands are for an additive F-CATE (no interactions between covariates). The heterogeneity variable (distance) was chosen after examining all continuous covariates; the null results for the other two are reported.
