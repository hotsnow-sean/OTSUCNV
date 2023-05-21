# This is a example to help use OTSUCNV.

from binning import binning
from otsu_cnv import OTSUCNV

# params
bam_path = "path/..."
fa_path = {
    "21": "path/...",
    "20": "path/...",
}
bp_per_bin = 1000

# generate RD profile
data = binning(bam_path, fa_path, bp_per_bin)

# OTSUCNV
CNVs = OTSUCNV(data["21"], bp_per_bin=bp_per_bin)
print(CNVs)

"""
You can detect CNVs on chromosomes corresponding to reference sequence names that have appeared in parameter.
eg:
OTSUCNV(data["20"], bp_per_bin=bp_per_bin)
"""
