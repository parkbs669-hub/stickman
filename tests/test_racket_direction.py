from pathlib import Path
import sys
import math
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stickman_racket import grip_direction, RacketDirectionTracker


def sample(left=False):
    lm = [SimpleNamespace(x=.5, y=.5, z=0, v=1) for _ in range(33)]
    elbow, wrist, pinky, index = (13,15,17,19) if left else (14,16,18,20)
    lm[elbow].x, lm[elbow].y = .2, .3
    lm[wrist].x, lm[wrist].y = .3, .4
    lm[pinky].x, lm[pinky].y = .32, .42
    lm[index].x, lm[index].y = .37, .38
    return lm


class RacketTests(unittest.TestCase):
    def test_racket_points_up_despite_forearm_pointing_down(self):
        dx, dy = grip_direction(sample(), 360, 640)
        self.assertGreater(dx, 0)
        self.assertLess(dy, 0)
        self.assertAlmostEqual(math.degrees(math.atan2(dy,dx)), -54.8879888, places=4)

    def test_left_grip_uses_left_landmarks_without_extra_flip(self):
        self.assertEqual(grip_direction(sample(True), 360, 640, False), grip_direction(sample(),360,640,True))

    def test_collapsed_fingers_are_not_trusted(self):
        lm = sample()
        lm[20].x, lm[20].y = lm[18].x+.0001, lm[18].y
        self.assertIsNone(grip_direction(lm,360,640))

    def test_low_visibility_is_rejected(self):
        lm = sample(); lm[20].v = .1
        self.assertIsNone(grip_direction(lm,360,640))

    def test_short_gap_holds_then_falls_back(self):
        tracker = RacketDirectionTracker(24)
        direction, _ = tracker.update(sample(),360,640)
        lm = sample();lm[20].v=0
        new, source = tracker.update(lm,360,640)
        self.assertEqual(new,direction)
        self.assertEqual(source,'held_grip_estimate')
        for _ in range(10): new, source = tracker.update(lm,360,640)
        self.assertEqual(source,'forearm_fallback')

    def test_wrap_around_uses_short_rotation(self):
        tracker = RacketDirectionTracker(24)
        tracker.angle=math.radians(179)
        lm=sample();lm[20].x=.27;lm[20].y=.419
        new,_=tracker.update(lm,360,640)
        self.assertLess(new[0], -.95)

    def test_pose_loss_resets_direction(self):
        tracker=RacketDirectionTracker(24);tracker.update(sample(),360,640)
        self.assertEqual(tracker.update(None,360,640),(None,'missing_pose'))
        self.assertIsNone(tracker.angle)


if __name__=='__main__':
    unittest.main(verbosity=2)
