"""T5d: weak labels, season windows, truth from the hand-check sheet, and a training smoke test (synthetic)."""
import numpy as np
import pandas as pd
import pytest

from src import vegetation_dl as vd

IG = vd.IGNORE


def test_window_labels_mark_stable_trees_tree_loss_and_other():
    start = np.array([1, 1, 1, 1, 2, 0, IG])
    end = np.array([1, 4, 2, 0, 1, 0, 1])
    assert vd.window_labels(start, end).tolist() == [1, 2, 2, 0, 0, 0, IG]               # tree->water is 'other', not loss


def test_window_indices_cover_four_consecutive_seasons_and_reject_impossible_windows():
    assert vd.window_indices(2018) == [0, 1, 2, 3] and vd.window_indices(2022) == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        vd.window_indices(2023)


def test_truth_counts_tree_loss_and_drops_unsure_answers():
    sheet = pd.DataFrame(dict(id=["T1", "T2", "T3", "T4", "T5"], trees_in_2023=["Yes", "Yes", "No", "Unsure", "Yes"],
                              now_covers=["Bare ground", "Trees", "Built-up", "Built-up", "Unsure"]))
    t = vd.truth_from_sheet(sheet)
    assert t.set_index("id").truth.to_dict() == {"T1": 1, "T2": 0, "T3": 0}


def test_auc_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(14), np.zeros(24)].astype(int); s = rng.normal(size=38) + 1.5 * y
    auc, (lo, hi) = vd.auc_ci(s, y, n_boot=500)
    assert lo <= auc <= hi and auc > 0.7


def test_training_on_several_windows_runs_and_reduces_the_loss():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    Xs = [rng.normal(size=(4, 64, 64)).astype("float32") for _ in range(2)]
    Ys = [np.where(X[0] > 0, 1, 0).astype("uint8") for X in Xs]                           # learnable rule on the first channel
    model, losses = vd.fit_windows(Xs, Ys, [(0, 0)], patch=64, steps=40, batch_size=4, seed=0, device="cpu")
    assert len(losses) > 10 and np.mean(losses[-5:]) < np.mean(losses[:5])


def test_loss_rich_patches_are_drawn_about_half_the_time():
    Ys = [np.zeros((256, 256), "uint8")]; Ys[0][:64, :64] = 2                              # loss only in the top-left patch
    wins = [(0, 0), (0, 128), (128, 0), (128, 128)]
    rich = vd.loss_rich_windows(Ys, wins, patch=128, min_share=0.01)
    assert rich == [(0, (0, 0))]
    rng = np.random.default_rng(0)
    draws = [vd.draw(Ys, wins, rich, rng, rich_share=0.5) for _ in range(2000)]
    share = np.mean([d == (0, (0, 0)) for d in draws])
    assert 0.5 < share < 0.7                                                            # half forced + a quarter of the rest


def test_inverse_frequency_weights_make_the_rare_class_heaviest():
    Y = np.zeros((128, 128), "uint8"); Y[:10, :10] = 2; Y[50:90, :] = 1
    w = vd.inverse_frequency_weights([Y], [(0, 0)], patch=128, n_classes=3)
    assert w[2] > w[1] > w[0] and np.isclose(w.mean(), 1.0)
