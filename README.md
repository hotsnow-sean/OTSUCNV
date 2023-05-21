<h1 align="center">
OTSUCNV
</h1>
<p align="center">
An Adaptive Segmentation and OTSU-based Anomaly Classification Method for CNV Detection Using NGS Data
</p>

<p align="center">
<a href="https://github.com/hotsnow-sean/OTSUCNV/example.py">Example</a> | <a href="#">Method</a>
</p>

<br>
<br>

## Dependency Package

- numpy
- pandas
- scikit-learn
- biopython
- pysam

> Require `python` >=v3.9

## Usage

```python
from binning import binning
from otsu_cnv import OTSUCNV

# params
bam_path = "path/...XXX.bam"
fa_path = {
    "21": "path/...XXX.fa",
    "20": "path/...XXX.fa",
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
```

## Input File

- Sequence alignment file in Bam format. (`.bam`)
- Reference sequence file in fasta format. (`.fa` or `.fasta`)

## Output Format

```python
# pd.Dataframe, eg:
"""
| start |  end  | type |
| 00001 | 11000 | gain |
| 21001 | 22000 | loss |
|  ...  |  ...  | .... |
"""
```
