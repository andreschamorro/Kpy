import argparse
import sys
import os
import gzip
from datetime import datetime
import json
import time
import math
import logging
import kpylib
import collections
import numpy as np
from tqdm import tqdm

def _parse_args():
    desc = "Calculate k-mer from cumulative relative entropy of all genomes"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("filenames", nargs="+", type=str, help="genome files in fasta format")
    parser.add_argument("-ke", "--kend", required=True, type=int, help="last k-mer")
    parser.add_argument("-kf", "--kfrom", default=4, type=int, help="Calculate from k-mer")
    parser.add_argument("-t", "--thread", type=int, default=1)
    parser.add_argument("-o", "--output", type=str, help="output filename")
    parser.add_argument("--logs_dir", type=str, default='.', help="Logs directory")
    parser.add_argument('--save', default='.', help='Directory to save model')

    return parser.parse_args()

def _create_outs_dir(parent_dir) -> str:
    """Standarized formating of outs dirs.
    Args:
        options (Options): information about the projects name.
    Returns:
        str: standarized logdir path.
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    outs_dir = os.path.join(parent_dir, "outs_dir-{}".format(now))
    # create file handler which logs even debug messages
    os.makedirs(f'{outs_dir}', exist_ok=True)
    return outs_dir 

def _get_logger(module_name, log_file):
    
    # create logger with 'module_name'
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # create file handler which logs even debug messages
    os.makedirs(f'{os.path.dirname(log_file)}', exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] : %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    verbose_formatter = logging.Formatter("[%(asctime)s.%(msecs)03d - %(levelname)s - %(process)d:%(thread)d - %(filename)s - %(funcName)s:%(lineno)d] %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

    fh.setFormatter(verbose_formatter)
    ch.setFormatter(verbose_formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def cal_e(count):
    result = 0
    total = count.sum
    for key in count.keys():
        q = count[key]/total
        s = math.log(total, 2) - math.log(count[key], 2)
        result += q*s

    return result

def cal_re(a0, a1, a2):
    """ Calculate Relative Entropy (RE)
        f' = fl * fr / fb
        example: f' of mer 'ATTTGCA' 
                f  is frequency of ATTTGCA from a0
                fl is frequency of ATTTGC- from a1
                fr is frequency of -TTTGCA from a1
                fb is frequency of -TTTGC- from a2
        RE = f * log2(f/f')
    Args:
        a0 Kmercount kmer
        a1 Kmercount kmer - 1
        a2 Kmercount kmer - 2

    Returns: float

    """
    result = 0
    rfactor = a0.sum
    lfactor = math.log((a1.sum ** 2) / (a2.sum * rfactor), 2)
    for key in a0.keys():
        realf = a0[key]
        left = a1[key[0:-1]]
        right = a1[key[1:]]
        below = a2[key[1:-1]]
        if 0 not in (left, right, below):
            expectf = left * right / below
            result += max(0, realf / rfactor * (math.log(realf / expectf, 2) + lfactor))
    return result

def stats(fsas, out_dir, kend, kfrom, **karg):

    a0 = None
    a1 = None
    a2 = None
    a0 = {fsa: None for fsa in fsas}
    a1 = {fsa: None for fsa in fsas}
    a2 = {fsa: None for fsa in fsas}
    ent = {fsa: [] for fsa in fsas}
    ent['kmer'] = []
    cre = {fsa: [] for fsa in fsas}
    cre['kmer'] = []
    acf = {}
    ofc = {}
    for kmer in tqdm(range(kend, kfrom-1, -1)):
        keys_acf = []
        keys_ofc = []
        for fsa in fsas:
            # CRE
            if a0[fsa] is None:
                a0[fsa] = kpylib.Kmercount(fsa, kmer, **karg)
                a1[fsa] = kpylib.Kmercount(fsa, kmer - 1, **karg)
                a2[fsa] = kpylib.Kmercount(fsa, kmer - 2, **karg)
            else:
                a0[fsa] = a1[fsa]
                a1[fsa] = a2[fsa]
                a2[fsa] = kpylib.Kmercount(fsa, kmer - 2, **karg)
            if kmer + 1 in cre[fsa]:
                cre[fsa].append(cal_re(a0[fsa], a1[fsa], a2[fsa]) + cre[kmer + 1])
            else:
                cre[fsa].append(cal_re(a0[fsa], a1[fsa], a2[fsa]))

            ent[fsa].append(cal_e(a0[fsa]))

            # ACF
            keys_acf.append(set(a0[fsa].keys()))
            # OFC
            keys_ofc.extend(list(a0[fsa].keys()))
        # CRE
        cre['kmer'].append(kmer)
        ent['kmer'].append(kmer)
        # ACF
        ccf = 0
        for pri_idx, key in enumerate(keys_acf):
            for sec_idx in range(pri_idx + 1, len(fsas)):
                ccf += len(keys_acf[pri_idx] & keys_acf[sec_idx])
        acf[kmer] = ccf/(len(fsas)-1)
        # OFC
        count_feature = list(collections.Counter((collections.Counter(keys_ofc).values())).values())
        lnp = np.log2(count_feature) - (kmer * 2)
        ofc[kmer] = np.sum(np.exp2(lnp) * lnp) * -1
    return cre, ent, acf, ofc

def main():
    args = _parse_args()
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(args.logs_dir, 'logs-{}'.format(now))
    logs = _get_logger('Kpy', log_file)

    logs.info("Create temp directory")
    outs_dir = _create_outs_dir(args.save)

    logs.info("Calculate Stats")
    cre, ent, acf, ofc = stats(args.filenames, outs_dir, **vars(args))

    logs.info("Save Stats")
    with open(os.path.join(outs_dir, "cre.json"), "w") as outfile:
        json.dump(cre, outfile)
    with open(os.path.join(outs_dir, "ent.json"), "w") as outfile:
        json.dump(ent, outfile)
    with open(os.path.join(outs_dir, "acf.json"), "w") as outfile:
        json.dump(acf, outfile)
    with open(os.path.join(outs_dir, "ofc.json"), "w") as outfile:
        json.dump(ofc, outfile)

if __name__ == "__main__":
    main()
