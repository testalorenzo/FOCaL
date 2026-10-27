"""
02_preprocess.py -- build data/processed/encode.npz for the ENCODE application.

One row per protein-coding gene on chr1-22 and chrX (GENCODE v29):
  Y  (n x 80)  H3K4me3 fold change over control in GM12878, log2(1 + fc), averaged in
               50-bp bins over [TSS - 2000, TSS + 2000), oriented so that the gene body
               is on the right (minus-strand profiles are reversed);
  A  (n,)      1 if a UCSC CpG island overlaps [TSS - 500, TSS + 500];
  X  (n x 5)   log(1 + TPM), log10 gene length, number of transcripts,
               log10(1 + distance to the nearest other gene's TSS), strand (1 if +).

The TSS of a gene is the transcript start shared by most of its APPRIS-principal
transcripts (fallback: all its basic transcripts), not the gene-level start of the GTF,
which lies more than 500 bp from the principal promoter for 20% of genes.

Run from the encode/ folder:  python3 02_preprocess.py
"""

import gzip
import re
from collections import Counter
import numpy as np
import pandas as pd
import pyBigWig

RAW, OUT, TEMP = 'data/raw/', 'data/processed/', 'data/temp/'
HALF_WIN, BIN, CPG_HALF = 2000, 50, 500          # window half-width, bin width, CpG overlap half-width (bp)
N_BINS = 2 * HALF_WIN // BIN                     # 80 bins
CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX']
counts = {}                                      # step -> number of genes, written to counts.txt


def attribute(key, info):
    """Value of `key "..."` in the attribute column of a GTF line."""
    return re.search(key + r' "([^"]+)"', info).group(1)


# ---------------------------------------------------------------------------
# 1. Genes and transcripts from the GTF (only "gene" and "transcript" lines are used)
# ---------------------------------------------------------------------------
genes, transcripts = [], []
with gzip.open(RAW + 'gencode.v29.basic.annotation.gtf.gz', 'rt') as f:
    for line in f:
        if line[0] == '#':
            continue
        chrom, _, feature, start, end, _, strand, _, info = line.rstrip('\n').split('\t')
        if feature == 'gene':
            genes.append((attribute('gene_id', info), chrom, int(start), int(end), strand,
                          attribute('gene_type', info)))
        elif feature == 'transcript':
            tss = int(start) if strand == '+' else int(end)
            transcripts.append((attribute('gene_id', info), tss, 'appris_principal' in info))

genes = pd.DataFrame(genes, columns=['gene_id', 'chrom', 'start', 'end', 'strand', 'gene_type'])
transcripts = pd.DataFrame(transcripts, columns=['gene_id', 'tss', 'principal'])
genes['gene_level_tss'] = np.where(genes['strand'] == '+', genes['start'], genes['end'])  # 1-based
counts['genes in GTF (all types, all contigs)'] = len(genes)

# ---------------------------------------------------------------------------
# 2. Protein-coding genes on the main chromosomes, with their transcript-based TSS
# ---------------------------------------------------------------------------
pc = genes[(genes['gene_type'] == 'protein_coding') & genes['chrom'].isin(CHROMS)].copy()
counts['protein-coding on chr1-22, chrX'] = len(pc)


def principal_tss(tx):
    """Most common start among the APPRIS-principal transcripts (ties: first in annotation order)."""
    pool = tx[tx['principal']] if tx['principal'].any() else tx
    return Counter(pool['tss']).most_common(1)[0][0]


pc['tss'] = pc['gene_id'].map(transcripts.groupby('gene_id').apply(principal_tss))
pc = pc[pc['tss'].notna()].copy()
pc['tss'] = pc['tss'].astype(int)
counts['with a transcript-based TSS'] = len(pc)
moved = (pc['tss'] - pc['gene_level_tss']).abs()
print(f'analysis TSS differs from the gene-level TSS by > 500 bp for {(moved > 500).sum()} genes '
      f'({(moved > 500).mean():.3f}), by > 2 kb for {(moved > 2000).sum()}')

# ---------------------------------------------------------------------------
# 3. Covariates from the annotation
# ---------------------------------------------------------------------------
pc['log10_length'] = np.log10(pc['end'] - pc['start'] + 1)
pc['n_transcripts'] = pc['gene_id'].map(transcripts.groupby('gene_id').size()).fillna(0).astype(int)
pc['strand_plus'] = (pc['strand'] == '+').astype(int)

# Distance from the gene's TSS to the nearest gene-level TSS of any OTHER gene on the chromosome.
dist = pd.Series(np.nan, index=pc.index)
for chrom, sub in pc.groupby('chrom'):
    others = np.sort(genes.loc[genes['chrom'] == chrom, 'gene_level_tss'].values)
    for i, tss, own in zip(sub.index, sub['tss'], sub['gene_level_tss']):
        d = np.abs(others - tss).astype(float)
        d[np.searchsorted(others, own)] = np.inf   # drop one copy of the gene's own TSS
        dist[i] = d.min()
pc['log10_dist_tss'] = np.log10(1 + dist)
print(f'genes sharing a TSS position with another gene: {int((dist == 0).sum())}')

# ---------------------------------------------------------------------------
# 4. Expression: log(1 + TPM) from the RSEM gene quantification (ids matched without version)
# ---------------------------------------------------------------------------
rna = pd.read_csv(RAW + 'rnaseq_GM12878_gene_quant.tsv', sep='\t', usecols=['gene_id', 'TPM'])
rna = rna[rna['gene_id'].str.startswith('ENSG')]
rna['gene_id'] = rna['gene_id'].str.replace(r'\.\d+$', '', regex=True)
tpm = rna.drop_duplicates('gene_id').set_index('gene_id')['TPM']
pc['TPM'] = pc['gene_id'].str.replace(r'\.\d+$', '', regex=True).map(tpm)
pc = pc[pc['TPM'].notna()].copy()
counts['with an RNA-seq quantification'] = len(pc)
pc['log1p_TPM'] = np.log1p(pc['TPM'])

# ---------------------------------------------------------------------------
# 5. Treatment: a CpG island overlapping [TSS - 500, TSS + 500]
# ---------------------------------------------------------------------------
cpg = pd.read_csv(RAW + 'cpgIslandExt_hg38.txt.gz', sep='\t', header=None, usecols=[1, 2, 3],
                  names=['chrom', 'start', 'end'])             # UCSC: 0-based, half-open [start, end)
pc['tss0'] = pc['tss'] - 1                                      # GTF is 1-based, bigWig/UCSC are 0-based
A = pd.Series(0, index=pc.index)
for chrom, sub in pc.groupby('chrom'):
    isl = cpg[cpg['chrom'] == chrom]
    s, e = isl['start'].values, isl['end'].values
    for i, tss0 in zip(sub.index, sub['tss0']):
        A[i] = int(((s < tss0 + CPG_HALF + 1) & (e > tss0 - CPG_HALF)).any())
pc['A'] = A
print(f'treated (CpG island at the promoter): {A.sum()} of {len(pc)} ({A.mean():.3f})')

# ---------------------------------------------------------------------------
# 6. Outcome: H3K4me3 fold change in 50-bp bins, gene body to the right
# ---------------------------------------------------------------------------
bw = pyBigWig.open(RAW + 'H3K4me3_GM12878_fc.bigWig')
inside = (pc['tss0'] - HALF_WIN >= 0) & (pc['tss0'] + HALF_WIN <= pc['chrom'].map(bw.chroms()))
pc = pc[inside].copy()
counts['with the full 4-kb window inside the chromosome'] = len(pc)

Y = np.zeros((len(pc), N_BINS))
n_missing = 0
for k, (chrom, tss0, strand) in enumerate(zip(pc['chrom'], pc['tss0'], pc['strand'])):
    vals = bw.stats(chrom, int(tss0 - HALF_WIN), int(tss0 + HALF_WIN), type='mean', nBins=N_BINS, exact=True)
    n_missing += sum(v is None for v in vals)
    vals = np.array([0.0 if v is None else v for v in vals])    # no signal -> 0
    if strand == '-':
        vals = vals[::-1]
    Y[k] = np.log2(1.0 + vals)
bw.close()
counts['final sample size n'] = len(pc)
print(f'bins with no bigWig value (set to 0): {n_missing}')
positions = np.arange(N_BINS) * BIN - HALF_WIN + BIN / 2        # bin centres relative to the TSS

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
X_cols = ['log1p_TPM', 'log10_length', 'n_transcripts', 'log10_dist_tss', 'strand_plus']
np.savez(OUT + 'encode.npz', Y=Y, A=pc['A'].values, X=pc[X_cols].values.astype(float),
         X_cols=np.array(X_cols), gene_id=pc['gene_id'].values.astype(str),
         chrom=pc['chrom'].values.astype(str), tss=pc['tss'].values,
         gene_level_tss=pc['gene_level_tss'].values, strand=pc['strand'].values.astype(str),
         positions=positions, TPM=pc['TPM'].values)
pc.to_csv(TEMP + 'genes_table.csv', index=False)
with open(OUT + 'counts.txt', 'w') as f:
    for step, n in counts.items():
        f.write(f'{step}\t{n}\n')
    f.write(f'treated\t{int(pc["A"].sum())}\ntreated share\t{pc["A"].mean():.4f}\n')

for step, n in counts.items():
    print(f'{step:55s} {n:>8d}')
tercile = pd.qcut(pc['log1p_TPM'], 3, labels=['low', 'mid', 'high'])
print('treated share by expression tercile:', pc.groupby(tercile, observed=True)['A'].mean().round(3).to_dict())
print(f'saved {OUT}encode.npz  (Y {Y.shape})')
