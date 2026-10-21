"""T4: land-use U-Net (Sentinel-2 -> 6 classes, weak labels from Esri annual LULC)."""
import numpy as np
import pytest
import torch
from torch import nn

from src import landuse as lu
from src import validation_points as vp

H, W = 1100, 1000   # synthetic canvas for split tests


# ---------------------------------------------------------------- labels
def test_class_order_and_mapping_are_shared_with_the_validation_sample():
    assert lu.CLASSES == vp.CLASSES and lu.IO_TO_CLASS is vp.IO_TO_CLASS
    assert len(lu.CLASSES) == 6


def test_label_mapping_is_total_and_ignores_unusable_codes():
    io = np.array([[0, 1, 2, 4], [5, 7, 8, 9], [10, 11, 3, 6]], dtype="uint8")
    out = lu.map_labels(io)
    assert out.dtype == np.uint8
    expect = {0: lu.IGNORE, 1: 0, 2: 1, 4: 0, 5: 2, 7: 4, 8: 5, 9: lu.IGNORE, 10: lu.IGNORE, 11: 3, 3: lu.IGNORE, 6: lu.IGNORE}
    for code, cls in expect.items():
        assert (out[io == code] == cls).all(), code


# ---------------------------------------------------------------- spatial split
def test_split_is_spatially_disjoint_and_patches_stay_inside_blocks():
    tr, va = lu.split_patches((H, W), patch=128, stride=64, block=512, val_frac=0.3, seed=0)
    assert len(tr) > 20 and len(va) > 5
    canvas_tr = np.zeros((H, W), bool); canvas_va = np.zeros((H, W), bool)
    for (r, c) in tr:
        assert r + 128 <= H and c + 128 <= W
        assert r // 512 == (r + 127) // 512 and c // 512 == (c + 127) // 512       # inside one block
        canvas_tr[r:r + 128, c:c + 128] = True
    for (r, c) in va:
        assert r // 512 == (r + 127) // 512 and c // 512 == (c + 127) // 512
        canvas_va[r:r + 128, c:c + 128] = True
    assert not (canvas_tr & canvas_va).any()                                       # no pixel is in both splits
    assert lu.split_patches((H, W), 128, 64, 512, 0.3, 0) == (tr, va)              # deterministic


def test_split_changes_with_seed():
    a = lu.split_patches((H, W), 128, 64, 512, 0.3, seed=0)[1]
    b = lu.split_patches((H, W), 128, 64, 512, 0.3, seed=1)[1]
    assert a != b


# ---------------------------------------------------------------- statistics / weights
def test_standardisation_stats_use_only_the_given_windows():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(3, 300, 300)).astype("float32")
    X[:, 150:, :] = 100.0                                                          # bottom half: a very different region
    mean, std = lu.band_stats(X, [(0, 0), (0, 100)], patch=100)                    # windows only in the top half
    assert mean == pytest.approx([5, 5, 5], abs=0.1) and std == pytest.approx([2, 2, 2], abs=0.1)


def test_stats_ignore_nan_and_standardise_nodata_to_zero():
    X = np.array([[[1.0, np.nan], [3.0, 5.0]]], dtype="float32")
    mean, std = lu.band_stats(X, [(0, 0)], patch=2)
    Z = lu.standardise(X, mean, std)
    assert np.isfinite(Z).all() and Z[0, 0, 1] == 0.0
    assert mean[0] == pytest.approx(3.0)


def test_class_weights_are_finite_and_inverse_to_frequency():
    Y = np.zeros((100, 100), "uint8"); Y[:10] = 1; Y[:2] = 2; Y[50:52, :20] = lu.IGNORE
    w = lu.class_weights(Y, [(0, 0)], patch=100, n_classes=6)
    assert np.isfinite(w).all() and (w > 0).all()
    assert w[2] > w[1] > w[0]                                                      # rarer class, larger weight
    assert w[5] == pytest.approx(w.max())                                          # absent class: capped, not infinite


# ---------------------------------------------------------------- model + training
def test_model_output_shape_and_size():
    m = lu.build_model(in_ch=10, n_classes=6)
    y = m(torch.zeros(2, 10, 64, 64))
    assert y.shape == (2, 6, 64, 64)
    assert sum(p.numel() for p in m.parameters()) < 25e6


def test_loss_ignores_the_ignore_label():
    logits = torch.randn(1, 6, 8, 8)
    y = torch.full((1, 8, 8), lu.IGNORE, dtype=torch.long)
    loss = nn.CrossEntropyLoss(ignore_index=lu.IGNORE)(logits, y)
    assert torch.isnan(loss)                                                       # all-ignored batch is NaN: fit() must guard it
    X = np.random.default_rng(0).normal(size=(10, 128, 128)).astype("float32")
    Yall = np.full((128, 128), lu.IGNORE, "uint8")
    model = lu.build_model(10, 6)
    losses = lu.fit(model, X, Yall, [(0, 0)], patch=64, steps=3, batch_size=2, device="cpu", augment=False)
    assert all(np.isfinite(losses))                                                # fit() skips batches with no valid pixel


def test_tiny_model_can_overfit_a_handful_of_patches():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # 4 patches whose class is a simple function of the first bands, so a working loop must reach high accuracy
    X = rng.normal(size=(10, 128, 128)).astype("float32")
    Y = np.argmax(X[:6], axis=0).astype("uint8")
    wins = [(0, 0), (0, 64), (64, 0), (64, 64)]
    model = lu.build_model(10, 6)
    losses = lu.fit(model, X, Y, wins, patch=64, steps=120, batch_size=4, lr=3e-3, device="cpu", augment=False, seed=0)
    assert losses[-1] < 0.6 * losses[0]
    # Evaluate on the geometry that was trained (64 px patches). Predicting the stitched 128 px image instead gave
    # 47 percent: on pure noise the net memorises where things sit inside each patch, so a new surrounding breaks
    # it (diagnosed: train- and eval-mode accuracy both 96 percent, per-patch predict_map 96 percent).
    # Tiling correctness is covered separately by test_tiled_prediction_has_no_seams_*.
    pred = np.zeros_like(Y)
    for r, c in wins:
        pred[r:r + 64, c:c + 64] = lu.predict_map(model, X[:, r:r + 64, c:c + 64], tile=64, margin=0, device="cpu")
    assert (pred == Y).mean() > 0.9


def test_fit_is_deterministic_on_cpu():
    X = np.random.default_rng(1).normal(size=(10, 128, 128)).astype("float32")
    Y = np.argmax(X[:6], axis=0).astype("uint8")
    out = []
    for _ in range(2):
        torch.manual_seed(5)
        out.append(lu.fit(lu.build_model(10, 6), X, Y, [(0, 0), (0, 64)], patch=64, steps=5, batch_size=2,
                          device="cpu", augment=True, seed=7))
    assert out[0] == out[1]


# ---------------------------------------------------------------- tiled inference
class _FirstSixBands(nn.Module):
    """A fixed 'model': logits are the first six input bands, so the exact answer is known."""
    def forward(self, x):
        return x[:, :6]


def test_tiled_prediction_has_no_seams_and_matches_direct_computation():
    X = np.random.default_rng(3).normal(size=(10, 301, 257)).astype("float32")      # awkward size on purpose
    direct = np.argmax(X[:6], axis=0).astype("uint8")
    for tile, margin in [(128, 16), (96, 24), (400, 32)]:
        pred = lu.predict_map(_FirstSixBands(), X, tile=tile, margin=margin, device="cpu")
        assert pred.shape == (301, 257) and pred.dtype == np.uint8
        assert np.array_equal(pred, direct), (tile, margin)


# ---------------------------------------------------------------- metrics
def test_confusion_and_metrics_on_a_hand_checked_example():
    ref = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, lu.IGNORE])
    pred = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 0])
    c = lu.confusion(pred, ref, n_classes=3)
    assert c.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 3]]                           # rows = reference, cols = prediction
    m = lu.metrics(c)
    assert m["overall_accuracy"] == pytest.approx(7 / 9)
    assert m["recall"] == pytest.approx([2 / 3, 1.0, 3 / 4])
    assert m["precision"] == pytest.approx([2 / 3, 2 / 3, 1.0])
    assert m["iou"][0] == pytest.approx(2 / 4)


# ---------------------------------------------------------------- real inputs
@pytest.fixture(scope="module")
def inputs():
    return lu.load_inputs("s2_2023", io_year=2023)


def test_real_inputs_are_on_the_grid_and_labels_cover_the_scene(inputs):
    X, Y = inputs["X"], inputs["Y"]
    assert X.shape[0] == 10 and X.shape[1:] == Y.shape == (2895, 2424)
    assert (Y != lu.IGNORE).mean() > 0.995                                          # almost every pixel has a usable label
    assert np.isfinite(X).mean() > 0.995
    assert np.nanmin(X) >= 0 and np.nanmax(X) < 3.0                                  # reflectance scale after /10000


def test_real_split_leaves_every_class_in_validation(inputs):
    tr, va = lu.split_patches(inputs["Y"].shape, 128, 64, 512, 0.25, seed=0)
    counts = np.zeros(6)
    for (r, c) in va:
        y = inputs["Y"][r:r + 128, c:c + 128]
        counts += np.bincount(y[y != lu.IGNORE], minlength=6)
    assert (counts > 500).sum() >= 5                                                # at least five classes represented
    assert counts[4] > 50_000                                                       # built-up is well covered


# ---------------------------------------------------------------- process isolation (regression)
def _run_py(code):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))


def test_baseline_module_never_loads_torch():
    """Regression: LightGBM and PyTorch each ship their own libomp; loading both in one macOS process crashed
    the seed-0 run with SIGSEGV in libomp right after the U-Net finished. The LightGBM baseline must therefore
    run in a process that has never imported torch."""
    r = _run_py("import sys; import src.landuse_baseline as b; import src.landuse as l; "
                "assert 'torch' not in sys.modules, 'torch was imported'; print('ok')")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-400:]


def test_landuse_helpers_import_without_torch():
    r = _run_py("import sys; import src.landuse as l; assert 'torch' not in sys.modules; print('ok')")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-400:]
