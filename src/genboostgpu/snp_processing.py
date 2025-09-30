import cudf
import cupy as cp
from cuml.preprocessing import SimpleImputer

__all__ = [
    "filter_zero_variance",
    "impute_snps",
    "run_ld_clumping",
    "preprocess_genotypes",
    "filter_cis_window",
]

def filter_zero_variance(X, snp_ids, threshold=1e-8):
    """
    Removes SNPs with variance < threshold.
    """
    vars_ = X.var(axis=0)
    keep_idx = cp.where(vars_ > threshold)[0]
    return X[:, keep_idx], [snp_ids[i] for i in keep_idx.tolist()]


def impute_snps(X, strategy="most_frequent"):
    """
    Impute missing genotypes using cuML SimpleImputer.
    Returns CuPy ndarray.
    """
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(X)


def ld_func(r2_thresh):
    return int(100 / r2_thresh)


def run_ld_clumping(X, snp_pos, stat, r2_thresh=0.1, fnc=ld_func):
    """
    PLINK-like greedy LD clumping on GPU.

    Default for size is ld_func: 100 / r2_thresh.
    """
    n_snps = X.shape[1]
    snp_pos = cp.asarray(snp_pos)
    stat = cp.asarray(stat)

    # sort SNPs by descending |stat|
    order = cp.argsort(cp.abs(stat))[::-1]

    keep = []
    pruned = cp.zeros(n_snps, dtype=bool)

    for idx in order.tolist():
        if pruned[idx]:
            continue

        keep.append(idx)
        pos = snp_pos[idx]

        # restrict to nearby SNPs
        kb_window = fnc(r2_thresh)
        neighbor_mask = (cp.abs(snp_pos - pos) <= kb_window)
        neighbor_idx = cp.where(neighbor_mask)[0]

        # batch compute LD
        X_sel = X[:, [idx] + neighbor_idx.tolist()]
        R = cp.corrcoef(X_sel.T)
        r2 = R[0, 1:] ** 2

        # prune neighbors above r2 threshold
        prune_mask = r2 >= r2_thresh
        for j in cp.asarray(neighbor_idx)[prune_mask].tolist():
            pruned[j] = True

    return cp.asarray(keep)


def preprocess_genotypes(X, snp_ids, snp_pos, y,
                         var_thresh=1e-8, impute_strategy="most_frequent",
                         r2_thresh=0.1, fnc=ld_func):
    """
    Full preprocessing pipeline:
    1. Zero-variance filter
    2. Impute missing (default assumes hard calls; use mean for dosage-style genotypes)
    3. LD clumping with phenotype-informed stats
    """
    # Filter zero variance SNPs
    X, snp_ids = filter_zero_variance(X, snp_ids, threshold=var_thresh)
    snp_pos = cp.asarray(snp_pos)[cp.isin(cp.arange(len(snp_pos)),
                                          cp.asarray([snp_ids.index(i) for i in snp_ids]))]

    # Impute missing
    X = impute_snps(X, strategy=impute_strategy)

    # Association stats (correlation with y)
    stat = cp.abs(cp.corrcoef(X.T, y)[-1, :-1])

    # LD clumping
    keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=r2_thresh, fnc=fnc)

    # Final reduced matrix
    return X[:, keep_idx], [snp_ids[i] for i in keep_idx.tolist()]


def filter_cis_window(geno_arr, bim, chrom, pos: int, end: int = None,
                      window_size: int = 20_000, use_window: bool = False):
    """
    Select SNPs within a cis-window around a CpG/phenotype position.
    """
    # define cis window boundaries
    if not use_window:
        window_size = 0
    
    start = pos - window_size
    if end is None:
        end = pos
    end   = end + window_size

    # select SNPs in window
    mask = (bim.chrom.astype(str) == str(chrom)) & \
        (bim.pos >= start) & (bim.pos <= end)

    if not mask.any():
        return None, [], []

    snp_ids = bim.loc[mask, "snp"].tolist()
    snp_pos = bim.loc[mask, "pos"].tolist()
    snp_idx = bim.index[mask].to_numpy()

    return geno_arr[:, snp_idx], snp_ids, snp_pos
