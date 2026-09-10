"""Regression checks for near-camera shoe mirroring and real foot movement."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tennis_stickman_v12_4_upgraded.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("tennis_feet_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def render_foot(dx=0.0, offset=(0, 0), side="right"):
    offset = np.asarray(offset, dtype=float)
    canvas = np.full((320, 320, 3), 255, dtype=np.uint8)
    module.draw_shoe(canvas, np.array([160, 140]) + offset,
                     np.array([160, 144]) + offset,
                     np.array([160 + dx, 184]) + offset,
                     np.array([165, 60]) + offset,
                     48, False, side_key=side)
    return canvas


def foreground_center(canvas):
    y, x = np.where(np.any(canvas < 245, axis=2))
    return np.array([x.mean(), y.mean()])


class FootStabilityTests(unittest.TestCase):
    def setUp(self):
        module._shoe_blend_cache = {"left": None, "right": None}

    def test_vertical_foot_noise_does_not_mirror_shoe(self):
        centers = [foreground_center(render_foot(dx))
                   for dx in [-0.3, 0.0, 0.3, -0.2, 0.2, -0.1] * 3]
        jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        self.assertLess(float(jumps.max()), 2.0)

    def test_real_foot_lift_moves_shoe_immediately(self):
        initial = foreground_center(render_foot())
        lifted = foreground_center(render_foot(offset=(12, -28)))
        np.testing.assert_allclose(lifted - initial, [12, -28], atol=0.1)

    def test_angle_wrap_takes_short_path(self):
        directions = []
        for degrees in [179, -179]:
            angle = math.radians(degrees)
            direction, _, _ = module._stable_shoe_basis(
                np.array([math.cos(angle), math.sin(angle)]) * 40,
                np.array([0, 80]), 48, "right")
            directions.append(direction)
        self.assertGreater(float(np.dot(*directions)), 0.99)
        self.assertLess(directions[1][0], -0.99)

    def test_collapsed_heel_toe_is_finite_after_side_view(self):
        module._stable_shoe_basis(np.array([40, 0]), np.array([0, 80]), 48, "right")
        with np.errstate(all="raise"):
            direction, normal, _ = module._stable_shoe_basis(
                np.array([0, 0]), np.array([0, 80]), 48, "right")
            canvas = np.full((320, 320, 3), 255, dtype=np.uint8)
            module.draw_shoe(canvas, (160, 140), (160, 140), (160, 140),
                             (160, 60), 48, False, "right")
        self.assertTrue(np.isfinite(direction).all())
        self.assertTrue(np.isfinite(normal).all())
        self.assertTrue(np.any(canvas < 245))

    def test_left_foot_history_does_not_rotate_right_foot(self):
        reference = render_foot()
        module._shoe_blend_cache = {"left": None, "right": None}
        for _ in range(8):
            render_foot(dx=-100, side="left")
        np.testing.assert_array_equal(reference, render_foot())


if __name__ == "__main__":
    unittest.main()
