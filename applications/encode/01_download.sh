#!/bin/bash
# 01_download.sh -- raw data for the ENCODE application of FOCaL.
#
# Effect of a promoter CpG island on the H3K4me3 fold-change profile around the
# TSS of protein-coding genes in GM12878, with heterogeneity by expression.
#
# Four files, all GRCh38 / GENCODE v29. Each download is skipped if the file
# already exists (the raw data are never deleted or overwritten by this script).
# Run from the encode/ folder:  bash 01_download.sh
# Downloaded on 2026-10-08.

set -e
mkdir -p data/raw batch_logs

# 1. H3K4me3 ChIP-seq, GM12878, fold change over control, pooled replicates (ENCODE ENCFF927KAJ, 705,791,872 bytes).
if [ ! -f data/raw/H3K4me3_GM12878_fc.bigWig ]; then
  curl -L -o data/raw/H3K4me3_GM12878_fc.bigWig \
    "https://www.encodeproject.org/files/ENCFF927KAJ/@@download/ENCFF927KAJ.bigWig" \
    > batch_logs/dl_bigwig.log 2>&1
fi

# 2. RNA-seq gene quantification, GM12878, RSEM on GENCODE v29 (ENCODE ENCFF413MYB).
if [ ! -f data/raw/rnaseq_GM12878_gene_quant.tsv ]; then
  curl -L -o data/raw/rnaseq_GM12878_gene_quant.tsv \
    "https://www.encodeproject.org/files/ENCFF413MYB/@@download/ENCFF413MYB.tsv" \
    > batch_logs/dl_rna.log 2>&1
fi

# 3. GENCODE v29 basic annotation (GTF).
if [ ! -f data/raw/gencode.v29.basic.annotation.gtf.gz ]; then
  curl -L -o data/raw/gencode.v29.basic.annotation.gtf.gz \
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.basic.annotation.gtf.gz" \
    > batch_logs/dl_gencode.log 2>&1
fi

# 4. UCSC CpG islands, hg38 (table cpgIslandExt).
if [ ! -f data/raw/cpgIslandExt_hg38.txt.gz ]; then
  curl -L -o data/raw/cpgIslandExt_hg38.txt.gz \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz" \
    > batch_logs/dl_cpg.log 2>&1
fi

ls -l data/raw
