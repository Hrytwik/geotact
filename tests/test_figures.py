"""Chapter figures: crop window and map extent helpers (synthetic)."""
import numpy as np
from affine import Affine

from src import figures as fg


def test_crop_window_is_the_mask_bounding_box_plus_padding_clipped_to_the_grid():
    m = np.zeros((100, 80), bool)
    m[20:30, 10:50] = True
    assert fg.crop_window(m, pad=5) == (15, 35, 5, 55)
    assert fg.crop_window(m, pad=50) == (0, 80, 0, 80)


def test_extent_follows_the_affine_transform():
    tf = Affine(10.0, 0, 1000.0, 0, -10.0, 5000.0)
    left, right, bottom, top = fg.extent(tf, (2, 6, 1, 4))
    assert (left, right, bottom, top) == (1010.0, 1040.0, 4940.0, 4980.0)
