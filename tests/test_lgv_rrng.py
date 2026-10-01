import numpy as np

from genboostgpu.lgv.folds import make_balanced_folds, offset_seed, stable_seed
from genboostgpu.lgv.rrng import RRNG, r_sample, r_sample_int


def test_unif_rand_matches_r(fx):
    ref = np.loadtxt(fx("rng_runif_20250805.txt"))
    np.testing.assert_array_equal(RRNG(20250805).unif_rand(5), ref)


def test_sample_vector_matches_r(fx):
    ref = np.loadtxt(fx("rng_sample_123.txt")).astype(int)
    out = r_sample(RRNG(123), np.resize(np.arange(1, 6), 23))
    np.testing.assert_array_equal(out, ref)


def test_sample_int_permutation_matches_r(fx):
    ref = np.loadtxt(fx("rng_sampleint_2147483628.txt")).astype(int)
    np.testing.assert_array_equal(r_sample_int(RRNG(2147483628), 153), ref)


def test_make_balanced_folds_matches_r(fx):
    ref = np.loadtxt(fx("folds_117_5_987654321.txt")).astype(int)
    np.testing.assert_array_equal(make_balanced_folds(117, 5, 987654321), ref)


def test_stable_seed_matches_sealed_stage01_row():
    # feature_seed recorded in lgv-AA-caudate-20260823 task 3.
    assert stable_seed("lgv-AA-caudate-20260823", "caudate", "chr1:903932-904084",
                       "joint_features") == 90227333


def test_offset_seed_wraps_without_overflow():
    assert offset_seed(2147483620, 1009) == (2147483620 + 1009) % 2147483629
    assert offset_seed(1, -5) >= 1
