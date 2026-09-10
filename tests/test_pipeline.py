import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import tennis_stickman_v12_4_upgraded as app
import stickman_runtime as runtime
import stickman_launcher as launcher


def pose(x=0.5, v=0.7):
    return [SimpleNamespace(x=x, y=0.5, z=0, v=v) for _ in range(33)]


class PipelineTests(unittest.TestCase):
    def test_manual_impacts_bypass_detection_and_spacing(self):
        with patch.object(app, 'detect_impacts_from_tracknet', side_effect=AssertionError):
            for serve in [True, False]:
                points, method = app.choose_impacts('unused', 30, [pose()]*100, True, serve,
                                                   manual=[61, 60, 60], ball_track=True)
                self.assertEqual(points, [60, 61])
                self.assertEqual(method, 'manual')

    def test_invalid_manual_frame_is_not_clamped(self):
        with self.assertRaises(ValueError):
            app.choose_impacts('unused', 30, [pose()]*10, True, False, manual=[10])

    def test_no_midpoint_fabrication(self):
        points, method = app.choose_impacts('unused', 30, [None]*10, True, False)
        self.assertEqual((points, method), ([], 'none'))

    def test_bounded_gaps_only(self):
        frames = [None, pose(.40), None, pose(.50), None, None, None, pose(.55), None]
        self.assertEqual(app.fill_short_pose_gaps(frames, 2), 1)
        self.assertAlmostEqual(frames[2][0].x, .45)
        self.assertIsNone(frames[0])
        self.assertIsNone(frames[-1])
        self.assertIsNone(frames[5])

    def test_gaps_across_cut_not_interpolated(self):
        frames = [pose(.1), None, pose(.9)]
        self.assertEqual(app.fill_short_pose_gaps(frames, 3), 0)

    def test_cache_preserves_actual_confidence(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE/'tests') as directory:
            path = Path(directory)/'cache.json'
            runtime.save_pose_cache(path, {'source':'a'}, [pose(v=.13)])
            restored = runtime.load_pose_cache(path, {'source':'a'})
            self.assertAlmostEqual(restored[0][0].v, .13)
            self.assertIsNone(runtime.load_pose_cache(path, {'source':'b'}))

    def test_ensemble_keyword_contract(self):
        with patch.object(app, 'detect_impacts_from_audio', return_value=([], [])):
            self.assertEqual(app.detect_impacts_ensemble('unused', 30, 10, [pose()]*10, True, min_gap_sec=.2), [])

    def test_truncated_cache_is_a_miss(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE/'tests') as directory:
            path = Path(directory)/'cache.json'
            runtime.save_pose_cache(path, {'frame_count': 2}, [pose()])
            self.assertIsNone(runtime.load_pose_cache(path, {'frame_count': 2}))

    def test_launcher_paths_are_separate_arguments(self):
        command = launcher.build_command('C:/한글 폴더/a & b.mp4', 'a & b', 'C:/results',
                                         '.5','720','backhand',True,True,True,False)
        self.assertIn('C:/한글 폴더/a & b.mp4', command)
        self.assertIn('--two-handed', command)
        self.assertNotIn('--overwrite', command)

    def test_invalid_speed_and_output_name_fail_early(self):
        for name, speed in [('valid',0), ('../escape',1), ('valid',float('nan'))]:
            with self.assertRaises(ValueError):
                app.process_video('unused',name,speed=speed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
