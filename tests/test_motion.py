"""Regression tests for motion, grip placement, and audio impact edge cases."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tennis_stickman_v12_4_upgraded.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("tennis_motion_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pose(wrist_x=0.4):
    result = [SimpleNamespace(x=0.5, y=0.5, z=0.0, v=1.0) for _ in range(33)]
    result[module.R_SHOULDER] = SimpleNamespace(x=0.3, y=0.4, z=0.0, v=1.0)
    result[module.R_WRIST] = SimpleNamespace(x=wrist_x, y=0.6, z=0.0, v=1.0)
    return result


class MotionTests(unittest.TestCase):
    def test_confidence_survives_smoothing(self):
        points = pose()
        points[27].v = 0.13
        smoothed = module.PoseSmoother(30).apply(points)
        self.assertAlmostEqual(smoothed[27].v, 0.13)
        points[27].v = 0.89
        self.assertAlmostEqual(module.PoseSmoother(30).apply(points)[27].v, 0.89)

    def test_swapped_leg_keeps_its_confidence(self):
        previous, current = pose(), pose()
        for left, right in [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
            previous[left].x, previous[right].x = 0.2, 0.8
            current[left].x, current[right].x = 0.8, 0.2
            current[left].v, current[right].v = 0.15, 0.91
        corrected = module.correct_leg_swaps(current, previous)
        self.assertAlmostEqual(corrected[27].x, 0.2)
        self.assertAlmostEqual(corrected[27].v, 0.91)
        self.assertAlmostEqual(corrected[28].v, 0.15)

    def test_missing_pose_global_correction_is_safe(self):
        for points in ([], [None], [None] * 10):
            self.assertEqual(module.correct_leg_orientation_global(points), points)

    def test_supersampling_does_not_change_final_limb_width(self):
        original_ssaa = module.SSAA
        try:
            for factor in (1, 2, 3):
                module.SSAA = factor
                module.configure_thickness(720 * factor)
                self.assertEqual(module.LIMB_THICKNESS / factor, 6)
                self.assertEqual(module.HEAD_OUTLINE_THICKNESS / factor, 8)
        finally:
            module.SSAA = original_ssaa
            module.configure_thickness(720 * original_ssaa)

    def test_grip_placement_for_both_hands(self):
        left, right = (100, 100), (200, 100)
        self.assertEqual(module.get_grip_points(left, right, 1, 0, 100, True, True),
                         ((265, 100), right))
        self.assertEqual(module.get_grip_points(left, right, -1, 0, 100, False, True),
                         (left, (35, 100)))
        self.assertEqual(module.get_grip_points(left, right, 1, 0, 100, False, False),
                         (left, right))

    def test_filter_stability_without_tuning_change(self):
        filt = module.OneEuroFilter(30, 0.2, 0.02, 1.0)
        self.assertEqual([filt(0.0) for _ in range(10)], [0.0] * 10)
        step = [filt(1.0) for _ in range(60)]
        self.assertTrue(all(0 <= value <= 1 for value in step))
        self.assertEqual(step, sorted(step))

    def test_stationary_and_missing_poses_do_not_invent_impacts(self):
        for points in ([], [None] * 30, [pose() for _ in range(60)]):
            impacts, extensions = module.detect_impacts_arm_extension(points, True)
            self.assertEqual(impacts, [])
            self.assertEqual(len(extensions), len(points))
            with patch.object(module, "detect_impacts_audio_validated", return_value=[]):
                self.assertEqual(module.detect_impacts_ensemble("unused", 30, len(points), points, True), [])

    def test_impact_deduplication_uses_seconds(self):
        points = []
        for frame in range(160):
            movement = max(0, 1 - abs(frame - 40) / 10, 1 - abs(frame - 100) / 10)
            points.append(pose(0.4 + movement * 0.4))
        at_30, _ = module.detect_impacts_arm_extension(points, True, fps=30, min_gap_sec=1)
        at_120, _ = module.detect_impacts_arm_extension(points, True, fps=120, min_gap_sec=1)
        self.assertEqual(len(at_30), 2)
        self.assertEqual(len(at_120), 1)


class AudioTests(unittest.TestCase):
    def extract(self, samples, fail=False):
        paths = []

        def fake_ffmpeg(command, **kwargs):
            target = command[-1]
            paths.append(Path(target).parent)
            if fail:
                Path(target).write_bytes(b"incomplete")
                raise subprocess.CalledProcessError(1, command)
            with wave.open(target, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(44100)
                output.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
            return subprocess.CompletedProcess(command, 0)

        with patch.object(module.subprocess, "run", side_effect=fake_ffmpeg):
            result = module.detect_impacts_from_audio("sample.mp4", 30, 90)
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))
        return result

    def test_missing_audio_returns_pair_and_removes_partial_file(self):
        self.assertEqual(self.extract([], fail=True), ([], []))

    def test_empty_short_and_silent_audio(self):
        for size in (0, 100, 882, 1101, 44100):
            with self.subTest(samples=size):
                self.assertEqual(self.extract(np.zeros(size)), ([], []))

    def test_single_onset_maps_near_the_video_impact(self):
        samples = np.zeros(44100, dtype=np.int16)
        samples[22050:22491] = 20000
        frames, onsets = self.extract(samples)
        self.assertEqual(len(frames), 1)
        self.assertLessEqual(abs(frames[0] - 15), 1)
        self.assertEqual(len(onsets), len(frames))


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main(verbosity=2)
