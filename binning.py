import math
import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pysam
from Bio import SeqIO


@dataclass
class RDdata:
    pos: np.ndarray  # the index of bin
    RD: np.ndarray  # the RD of bin

    def __init__(self, num_of_bin: int) -> None:
        self.pos = np.arange(num_of_bin)
        self.RD = np.zeros(num_of_bin)


@dataclass()
class GCData:
    valid: np.ndarray
    GC: np.ndarray  # GC per thousand (GCcount / bp_per_bin * 1000)
    bp_per_bin: int

    def __init__(self, fasta, bp_per_bin: int) -> None:
        self.bp_per_bin = bp_per_bin

        num_of_bin = len(fasta) // bp_per_bin
        self.valid = np.full(num_of_bin, True)
        self.GC = np.full(num_of_bin, 0)

        it = iter(fasta)

        for i in range(num_of_bin):
            for _ in range(bp_per_bin):
                bp = next(it)
                if self.valid[i] and bp not in "AGCTagct":
                    self.valid[i] = False
                if bp in "GCgc":
                    self.GC[i] += 1
            self.GC[i] = round(self.GC[i] / bp_per_bin * 1000)

    def limit(self, length: int):
        self.valid = self.valid[:length]
        self.GC = self.GC[:length]


def binning(bam_path: str, fa_path: dict, bp_per_bin: int) -> Dict[str, RDdata]:
    """Divide the DNA sequence into bins according to a certain length and calculate the RD value of each bin.

    Parameters
    ----------
    bam_path: str
    The path of bam file.

    fa_path: Dict[str, str]
    The path of reference.

    bp_per_bin: int
    The length of each bin.

    Notes
    -----
    The name in `fa_path` should appear in the information corresponding to the bam file, otherwise an exception will be thrown.
    """
    RDs: Dict[str, RDdata] = {}
    GCs: Dict[str, GCData] = {}
    with pysam.AlignmentFile(bam_path, "rb", ignore_truncation=True) as samfile:
        refs = {
            key: value for key, value in fa_path.items() if key in samfile.references
        }
        if not refs:
            raise ValueError("Valid reference name is not found")

        # calculate GC
        for chr, fasta in refs.items():
            expect_len = samfile.get_reference_length(chr)
            num_of_bin = expect_len // bp_per_bin
            if isinstance(fasta, str):
                fasta = SeqIO.read(fasta, "fasta").seq
                actual_len = len(fasta)
                if expect_len > actual_len:
                    raise ValueError(
                        f"{chr}'s reference length is not matched! expect {expect_len}, actual {actual_len}"
                    )
                elif expect_len < actual_len:
                    warnings.warn(
                        f"{chr}'s reference length is not matched! expect {expect_len}, actual {actual_len}"
                    )
                GCs[chr] = GCData(fasta, bp_per_bin)
            else:
                raise ValueError("'fa_path': the key should be a str")
            RDs[chr] = RDdata(num_of_bin)
            GCs[chr].limit(num_of_bin)

        # count read
        for read in samfile:
            if read.is_secondary:
                continue
            idx = read.reference_start // bp_per_bin
            chr = read.reference_name
            if chr in refs and idx < len(RDs[chr].RD):
                RDs[chr].RD[idx] += 1

    def gc_correct(RD: np.ndarray, Gc: np.ndarray):
        bin_count = np.bincount(Gc)
        global_rd_ave = np.mean(RD)
        for i in range(len(RD)):
            if bin_count[Gc[i]] < 2:
                continue
            mean = np.mean(RD[Gc == Gc[i]])
            if not math.isclose(mean, 0.0):
                RD[i] *= global_rd_ave / mean
        return RD

    # correct RD (del N, fill zero, correct gc bias)
    for chr, rd in RDs.items():
        rd.RD = rd.RD[GCs[chr].valid]
        rd.pos = rd.pos[GCs[chr].valid]
        rd.RD /= bp_per_bin  # read count -> read depth
        rd.RD = gc_correct(rd.RD, GCs[chr].GC[GCs[chr].valid])

    return RDs
