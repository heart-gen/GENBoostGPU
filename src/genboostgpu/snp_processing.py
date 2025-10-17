import cudf
import cupy as cp
from cuml.preprocessing import SimpleImputer

__all__ = [
    "filter_zero_variance",
    "impute_snps",
    "run_ld_clumping",
    "preprocess_genotypes",
    "filter_cis_window",
    "_corr_with_y_streaming"
]

def filter_zero_variance(X, snp_ids, snp_pos=None, threshold=1e-8):
    """
    Removes SNPs with variance < threshold.
    """
    vars_ = X.var(axis=0)
    keep_idx = cp.where(vars_ > threshold)[0]
    X2 = X[:, keep_idx]
    ids2 = [snp_ids[i] for i in keep_idx.tolist()]
    pos2 = None if snp_pos is None else [snp_pos[i] for i in keep_idx.tolist()]
    return X2, ids2, pos2


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


def preprocess_genotypes(X, snp_ids, snp_pos, y, var_thresh=1e-8,
                         impute_strategy="most_frequent", r2_thresh=0.1,
                         batch_size=8192, fnc=ld_func):
    """
    Full preprocessing pipeline:
    1. Zero-variance filter
    2. Impute missing (default assumes hard calls; use mean for dosage-style genotypes)
    3. LD clumping with phenotype-informed stats
    """
    # Filter zero variance SNPs
    X, snp_ids, snp_pos = filter_zero_variance(X, snp_ids, snp_pos, threshold=var_thresh)

    # Impute missing
    X = impute_snps(X, strategy=impute_strategy)

    # Association stats (correlation with y)
    stat = _corr_with_y_streaming(X, y, batch_size)

    # LD clumping
    keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=r2_thresh, fnc=fnc)

    # Final reduced matrix
    return X[:, keep_idx], [snp_ids[i] for i in keep_idx.tolist()]


def index_cis_window(bim, chrom, pos: int, end: int = None,
                     window_size: int = 20_000, use_window: bool = False):
    """
    Return indices/ids/positions for SNPs within a cis-window.
    """
    if not use_window:
        window_size = 0
    start = pos - window_size
    if end is None:
        end = pos
    end = end + window_size

    # Select SNP window
    m = (bim.chrom.astype(str) == str(chrom)) & \
        (bim.pos >= start) & (bim.pos <= end)

    if m.sum() == 0:
        return [], [], []

    # Grab index positions (zero-based relative to bim)
    is_cudf = hasattr(bim, "to_pandas")
    if is_cudf:
        # cuDF
        sub = bim.loc[m]
        idx = sub.index.to_pandas().to_numpy()
        snp_ids = sub["snp"].to_pandas().tolist()
        snp_pos = sub["pos"].to_pandas().tolist()
    else:
        # pandas fallback
        idx = bim.index[m].to_numpy()
        snp_ids = bim.loc[m, "snp"].tolist()
        snp_pos = bim.loc[m, "pos"].tolist()
    return idx, snp_ids, snp_pos


def filter_cis_window(geno_arr, bim, chrom, pos: int, end: int = None,
                      window_size: int = 20_000, use_window: bool = False):
    """
    Select SNPs within a cis-window around a CpG/phenotype position.
    """
    # Index SNP window
    idx, snp_ids, snp_pos = index_cis_window(bim, chrom, pos, end, 
                                             window_size, use_window)

    if idx is None or len(idx) == 0:
        return None, [], []

    return geno_arr[:, idx], snp_ids, snp_pos


def count_snps_in_window(bim, chrom, pos: int, end: int = None,
                         window_size: int = 20_000, use_window: bool = False) -> int:
    """Fast count for stratified sampling—no genotype access."""
    idx, _, _ = index_cis_window(bim, chrom, pos, end, window_size, use_window)
    return int(len(idx))


def _corr_with_y_streaming(X, y, batch_size: int = 8192, eps: float = 1e-12):
    X = X.astype(cp.float32, copy=False)  # halve memory
    y = y.astype(cp.float32, copy=False).ravel()

    y = y - y.mean()
    y_norm = cp.linalg.norm(y) + eps

    chunks = []
    for j in range(0, X.shape[1], batch_size):
        Xi = X[:, j:j+batch_size]
        Xi = Xi - Xi.mean(axis=0)
        denom = (cp.linalg.norm(Xi, axis=0) * y_norm + eps)
        r = (Xi.T @ y) / denom            # Pearson r per SNP
        chunks.append(cp.abs(r))
    return cp.concatenate(chunks, axis=0)
