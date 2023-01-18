import os
import sys
import copy
import math
import shutil
import platform
import tempfile
import warnings
import subprocess
import collections

from scipy.stats import chi2
from scipy.spatial import distance
from scipy.special import loggamma
import math
import collections
import numpy as np
from tqdm import tqdm

class NoInput(Exception):
    """Do X and return a list."""
    pass

if platform.system() == 'Darwin':
    jellyfishpath = os.path.join(__location__, 'modules', "jellyfish-macosx")
elif platform.system() == 'Linux':
    jellyfishpath = os.path.join('/usr/bin', "jellyfish")
else:
    raise Exception("Windows are not supported.")

class JellyFishError(Exception):
    def __init__(self, error_message):
        self.message = f"JellyFish dumping error: {error_message}"


very_small_number = 1e-100

def evo_transform(dist, kmer):
    """
    follow 1. Fan H, Ives AR, Surget-Groba Y, Cannon CH. 
    An assembly and alignment-free method of phylogeny reconstruction from 
    next-generation sequencing data. BMC Genomics [Internet]. 
    2015 Jul 14 [cited 2020 Apr 17];16(1):522.
D = (-1/k) * log(distance)
    """
    j = very_small_number if dist <= 0 else dist
    return (-1/kmer) *  math.log(j)

def mash(u, v, kmer):
    """Do X and return a list."""
    j = 1 - distance.jaccard(u, v)
    j = very_small_number if j <= 0 else j
    return (-1/kmer) * math.log(2*j/(1+j))

def jsmash(u, v, kmer):
    """Do X and return a list."""
    j = 1 - distance.jensenshannon(u, v)
    j = very_small_number if j <= 0 else j
    return (-1/kmer) * math.log(2*j/(1+j))


def nCr(n, r):
    """
    Calculate combinatorial using gamma funcion for huge number
    """
    logncr = loggamma(n+1) - loggamma(r+1) - loggamma(n-r+1)
    return math.exp(logncr)

def jaccarddistp(u, v):
    """
    Do X and return a list.
    """
    m = len(u)
    pu = u.sum()/m
    pv = v.sum()/m

    degenerate = False

    expectation = ((pu*pv)/(pu+pv-pu*pv))

    j_obs = np.logical_and(u, v).sum() / np.logical_or(u, v).sum() - expectation

    if(pu == 1 or pv == 1 or u.sum() == len(u) or v.sum() == len(v)):
        warnings.warn("One or both input vectors contain only 1's.", Warning)
        degenerate = True

    if(pu == 0 or pv == 0 or u.sum() == 0 or v.sum() == 0):
        warnings.warn("One or both input vectors contain only 0's", Warning)
        degenerate = True

    if degenerate:
        return 1.0

    # tan = jaccard_mca_rcpp(px,py,m,j.obs,accuracy)
    # pvalue = tan$pvalue

    #   pvalue <- switch(error.type, lower = pvalue, average = pvalue/tan$accuracy,
    #                upper = pvalue + 1 - tan$accuracy)

    # return(
    #     list(
    #     statistics = j.obs,
    #     pvalue = pvalue,
    #     expectation = expectation,
    #     accuracy = 1 - tan$accuracy,
    #     error.type = error.type
    #     )
    # )
    # Compute p-value using an asymptotic approximation

    q = [pu*pv, pu+pv-2*pu*pv]
    qq = q[0] + q[1]
    sigma = q[0] * q[1] * (1-q[0]) / (qq ** 3)
    norm = math.sqrt(m) * j_obs / math.sqrt(sigma)
    return chi2.pdf(norm*norm, 1)

def euclidean_of_frequency(u, v):
    """
    euclidean distance of frequency
    """
    return distance.euclidean(u, v)

DISTANCE_FUNCTION = {
    'braycurtis': distance.braycurtis,
    'canberra': distance.canberra,
    'chebyshev': distance.chebyshev,
    'cityblock': distance.cityblock,
    'correlation': distance.correlation,
    'cosine': distance.cosine,
    'dice': distance.dice,
    'euclidean': distance.euclidean,
    'hamming': distance.hamming,
    'jaccard': distance.jaccard,
    'kulsinski': distance.kulsinski,
    'rogerstanimoto': distance.rogerstanimoto,
    'russellrao': distance.russellrao,
    'sokalmichener': distance.sokalmichener,
    'sokalsneath': distance.sokalsneath,
    'sqeuclidean': distance.sqeuclidean,
    'yule': distance.yule,
    'jensenshannon': distance.jensenshannon,
    'mash': mash,
    'jsmash' : jsmash,
    'jaccarddistp': jaccarddistp,
    'euclidean_of_frequency': euclidean_of_frequency
}

NUMERIC_DISTANCE = [
    distance.braycurtis,
    distance.canberra,
    distance.chebyshev,
    distance.cityblock,
    distance.correlation,
    distance.cosine,
    distance.euclidean,
    distance.sqeuclidean
]

BOOLEAN_DISTANCE = [
    distance.dice,
    distance.hamming,
    distance.jaccard,
    jaccarddistp,
    distance.kulsinski,
    distance.rogerstanimoto,
    distance.russellrao,
    distance.sokalmichener,
    distance.sokalsneath,
    distance.yule
]

PROB_DISTANCE = [
    distance.jensenshannon,
    euclidean_of_frequency
]

class Kmercount(collections.Counter):
    """Do X and return a list."""
    def __init__(self, fsa, k_mer, **karg):

        self.kmer = k_mer

        if 'thread' not in karg:
            karg['thread'] = 1
        if 'lower' not in karg:
            karg['lower'] = 1
        if 'bchashsize' not in karg: #hashsize for jellyfish bc step
            karg['bchashsize'] = '1G'
        if 'hashsize' not in karg: #hashsize for jellyfish count step
            karg['hashsize'] = '100M'
        if 'canonical' not in karg or not karg['canonical']:
            canonical = ''
        else:
            canonical = '-C'

        if not os.path.isfile(fsa):
            raise NoInput('input is missing')

        filebasename = os.path.basename(fsa)

        with tempfile.TemporaryDirectory() as tmpdirname:

            if 'fast' in karg and karg['fast']:
                # for genome with one step
                dumpdata = subprocess.getoutput("""
                    {0} count {8} -m {1} -s {3} -t {4} -o {5}.jf {6}
                    {0} dump -c -L {7} {5}.jf
                    """.format(
                        jellyfishpath,
                        self.kmer,
                        karg['bchashsize'],
                        karg['hashsize'],
                        karg['thread'],
                        os.path.join(tmpdirname, filebasename),
                        fsa,
                        karg['lower'],
                        canonical
                    )
                )
            else:
                dumpdata = subprocess.getoutput("""
                    {0} bc {8} -m {1} -s {2} -t {4} -o {5}.bc {6}
                    {0} count {8} -m {1} -s {3} -t {4} --bc {5}.bc -o {5}.jf {6}
                    {0} dump -c -L {7} {5}.jf
                    """.format(
                        jellyfishpath,
                        self.kmer,
                        karg['bchashsize'],
                        karg['hashsize'],
                        karg['thread'],
                        os.path.join(tmpdirname, filebasename),
                        fsa,
                        karg['lower'],
                        canonical
                    )
                )

        datadict = {}
        if not dumpdata.startswith("Bloom filter file is truncated"):
            try:
                for line in dumpdata.split('\n'):
                    dat = line.rstrip().split(' ')
                    datadict[dat[0]] = int(dat[1])
            except ValueError:
                raise JellyFishError("Bloom filter file is truncated.")
        else:
            raise JellyFishError("Bloom filter file is truncated.")
        super(Kmercount, self).__init__(datadict)

        # assign instance variable
        self.sum = sum(self.values())
        self.name = '.'.join(filebasename.split('.')[0:-1]).replace(' ', '_')

    def __repr__(self):
        return self.name

    def dist(self, other, dist_func, transform=False):
        """Do X and return a list."""
        a, b = self.norm(other)
        if dist_func is mash:
            dist = dist_func(a.astype(bool), b.astype(bool), self.kmer)
        elif dist_func is jsmash:
            dist = dist_func(a.astype(float)/a.sum(), b.astype(float)/b.sum(), self.kmer)
        elif dist_func in BOOLEAN_DISTANCE:
            print(a.astype(bool), b.astype(bool))
            dist = dist_func(a.astype(bool), b.astype(bool))
        elif dist_func in PROB_DISTANCE:
            dist = dist_func(a.astype(float)/a.sum(), b.astype(float)/b.sum())
        else:
            dist = dist_func(a, b)

        dist = very_small_number if math.isnan(dist) else dist
        if transform and dist_func not in [mash, jsmash] + NUMERIC_DISTANCE:
            dist = evo_transform(dist, self.kmer)
        
        return dist

    def norm(self, other):
        """Do X and return a list."""
        mers = list(self.keys())
        mers.extend(list(other.keys()))
        mers = list(set(mers))
        a = []
        b = []
        for mer in mers:
            a.append(self[mer])
            b.append(other[mer])
        return np.array(a), np.array(b)

def cal_cre(fsa, kend, kfrom, **karg):
    """ Calculate Cumulative Relative Entropy (CRE)
        CRE = sum(RE from kmer to infinite)
    Args:
        fsa genome file in fasta format
        kend the infinite kmer
        kfrom calculate to (defualt=4)

    Returns: dict(kmer: CRE)
.. moduleauthor:: Natapol Pornputtapong <natapol.p@chula.ac.th>

    """
    a0 = None
    a1 = None
    a2 = None
    result = {}
    for kmer in tqdm(range(kend, kfrom-1, -1)):
        if a0 is None:
            a0 = jf.Kmercount(fsa, kmer, **karg)
            a1 = jf.Kmercount(fsa, kmer - 1, **karg)
            a2 = jf.Kmercount(fsa, kmer - 2, **karg)
        else:
            a0 = a1
            a1 = a2
            a2 = jf.Kmercount(fsa, kmer - 2, **karg)
        if kmer + 1 in result:
            result[kmer] = cal_re(a0, a1, a2) + result[kmer + 1]
        else:
            result[kmer] = cal_re(a0, a1, a2)
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

def cal_acf(fsas, kkeys, **karg):
    """Calculate Average number of common features (ACF)

    Args:
        fsas (str):  genome file name(s).
        kmers (): a list of kmer to calculate.

    Kwargs:
        state (bool): Current state to be in.
        thread (int): Number of thread to calculate default 1
        lower (int): default 1
        bchashsize (str): hashsize for jellyfish bc step default '1G'
        hashsize (str): hashsize for jellyfish count step default '100M'
        canonical (bool): set canonical calculation

    Returns:
        dict(kmer: acf)

    Raises:
        AttributeError, KeyError

    A really great idea.  A way you might use me is

    >>> print public_fn_with_googley_docstring(name='foo', state=None)
    0

    BTW, this always returns 0.  **NEVER** use with :class:`MyPublicClass`.

    """
    n = len(fsas)
    if n >= 2:
        result = {}
        for kmer in tqdm(kmers):
            keys_array = []
            for fsa in fsas:
                keys_array.append(set(jf.Kmercount(fsa, kmer, **karg).keys()))
            ccf = 0
            for pri_idx, key in enumerate(keys_array):
                for sec_idx in range(pri_idx + 1, n):
                    ccf += len(keys_array[pri_idx] & keys_array[sec_idx])
            result[kmer] = ccf/(n-1)
        return result
    else:
        raise

def cal_ofc(fsas, kmer, **karg):
    """ Calculate shannon entropy of observed feature occurrences (OFC)
        ofc(l) = -sum(p ln p)
    Args:
        fsa a genome file
        k_mers a list of kmer to calculate

    Returns: float shannon entropy

    """
    result = {}
    
    #for kmer in tqdm(k_mers):
    keys = []
    for fsa in fsas:
        keys.extend(list(jf.Kmercount(fsa, kmer, **karg).keys()))
        
    count_feature = list(collections.Counter((collections.Counter(keys).values())).values())
    lnp = np.log2(count_feature) - (kmer * 2)
    result[kmer] = np.sum(np.exp2(lnp) * lnp) * -1

    return result
