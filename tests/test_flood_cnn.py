"""T6g CNN and tabular MLP (torch, CPU, tiny synthetic data): shapes, augmentation, balanced batches, normalisation, learning."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src import flood_cnn as fc


def synth_patches(n=240, c=3, size=9, pos_frac=0.25, seed=0):
    """Flood patches have a dip (low values) in the centre; others are noise. A CNN should find the dip."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(int)
    x = rng.normal(size=(n, c, size, size)).astype("float32")
    h = size // 2
    x[y == 1, 0, h - 1:h + 2, h - 1:h + 2] -= 3.0
    return x, y


def test_cnn_returns_one_logit_per_patch_for_any_channel_count():
    for c in (3, 16):
        out = fc.FloodCNN(c)(torch.zeros(5, c, 33, 33))
        assert out.shape == (5,)


def test_augment_only_flips_and_rotates_so_pixel_values_are_preserved():
    x = np.random.default_rng(0).normal(size=(6, 3, 9, 9)).astype("float32")
    a = fc.augment(x, np.random.default_rng(1))
    assert a.shape == x.shape
    for i in range(len(x)):
        assert np.allclose(np.sort(a[i].ravel()), np.sort(x[i].ravel()))
    again = fc.augment(x, np.random.default_rng(1))
    assert np.array_equal(a, again)                                        # same rng, same result


def test_augment_actually_changes_some_patches():
    x = np.random.default_rng(0).normal(size=(40, 2, 9, 9)).astype("float32")
    assert not np.array_equal(fc.augment(x, np.random.default_rng(3)), x)


def test_balanced_batch_is_half_flood_and_samples_flood_with_replacement():
    y = np.r_[np.ones(5), np.zeros(500)].astype(int)
    rng = np.random.default_rng(0)
    idx = fc.balanced_batch(y, 32, rng)
    assert len(idx) == 32 and y[idx].sum() == 16
    assert len(set(idx[y[idx] == 1])) <= 5                                 # only 5 flood samples exist; repeats are expected


def test_zscore_uses_the_given_patches_and_zero_fills_nan_without_dividing_by_zero():
    x = np.random.default_rng(0).normal(loc=np.array([5.0, -2.0, 7.0]).reshape(1, 3, 1, 1), scale=np.array([2.0, 0.5, 1.0]).reshape(1, 3, 1, 1), size=(50, 3, 9, 9)).astype("float32")
    x[:, 2] = 7.0                                                          # constant channel: std 0
    x[0, 0, 0, 0] = np.nan
    mean, std = fc.fit_zscore(x)
    z = fc.apply_zscore(x, mean, std)
    assert np.isfinite(z).all()
    assert abs(z[:, 0][np.isfinite(x[:, 0])].mean()) < 0.01 and z[:, 1].std() == pytest.approx(1.0, abs=0.01)
    assert (z[:, 2] == 0).all()
    assert z[0, 0, 0, 0] == 0.0                                            # NaN became 0 after z-scoring


def test_cnn_training_is_deterministic_for_a_seed_and_learns_the_dip():
    from sklearn.metrics import roc_auc_score
    x, y = synth_patches(seed=0)
    xt, yt = synth_patches(n=200, seed=1)
    a = fc.train_cnn(x, y, seed=0, steps=120, batch=32)
    b = fc.train_cnn(x, y, seed=0, steps=120, batch=32)
    pa, pb = fc.predict(a, xt), fc.predict(b, xt)
    assert np.allclose(pa, pb, atol=1e-5)
    assert roc_auc_score(yt, pa) > 0.9


def test_mlp_learns_a_tabular_signal():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)); y = (rng.random(400) < 0.25).astype(int); X[:, 0] += 2.0 * y
    Xt = rng.normal(size=(300, 6)); yt = (rng.random(300) < 0.25).astype(int); Xt[:, 0] += 2.0 * yt
    m = fc.train_mlp(X, y, seed=0, steps=200, batch=32)
    assert roc_auc_score(yt, fc.predict_tab(m, Xt)) > 0.85


def test_cv_scores_each_sample_once_from_a_model_that_never_saw_its_fold():
    from sklearn.metrics import roc_auc_score
    x, y = synth_patches(n=300, seed=2)
    folds = np.arange(300) % 3
    oof = fc.cv_cnn(x, y, folds, seeds=(0,), steps=80, batch=32)
    assert oof.shape == (300,) and np.isfinite(oof).all()
    assert roc_auc_score(y, oof) > 0.85


def test_seed_averaging_returns_the_mean_of_the_seed_scores():
    x, y = synth_patches(n=150, seed=3)
    folds = np.arange(150) % 3
    per_seed = fc.cv_cnn(x, y, folds, seeds=(0, 1), steps=40, batch=32, return_per_seed=True)
    avg = fc.cv_cnn(x, y, folds, seeds=(0, 1), steps=40, batch=32)
    assert per_seed.shape == (2, 150)
    assert np.allclose(avg, per_seed.mean(axis=0), atol=1e-6)
