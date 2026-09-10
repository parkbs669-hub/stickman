"""Estimate racket shaft direction from the grip, rather than extending the arm.

MediaPipe Pose's index/pinky knuckle axis is only an estimate of the grip axis.
Collapsed or occluded finger landmarks must not be treated as reliable angles.
"""
import math


def _confidence(point):
    return float(getattr(point, "v", getattr(point, "visibility", 1.0)))


def grip_direction(landmarks, width, height, is_right_handed=True):
    """Return a directed screen-space unit vector, or None for ambiguous fingers."""
    if landmarks is None or len(landmarks) < 23:
        return None
    pinky, index, wrist, elbow = (18, 20, 16, 14) if is_right_handed else (17, 19, 15, 13)
    p, i, w, e = (landmarks[j] for j in (pinky, index, wrist, elbow))
    if min(_confidence(p), _confidence(i), _confidence(w)) < 0.5:
        return None
    # Work in pixel space: normalized x/y have different scales in portrait video.
    dx, dy = (i.x-p.x)*width, (i.y-p.y)*height
    length = math.hypot(dx, dy)
    arm = math.hypot((w.x-e.x)*width, (w.y-e.y)*height)
    if not math.isfinite(length) or length < max(2.5, height*0.005, arm*0.07):
        return None
    return dx/length, dy/length


class RacketDirectionTracker:
    def __init__(self, fps):
        self.fps = float(fps)
        self.angle = None
        self.missing = 0

    def reset(self):
        self.angle = None
        self.missing = 0

    def update(self, landmarks, width, height, is_right_handed=True):
        if landmarks is None:
            self.reset()
            return None, "missing_pose"
        direction = grip_direction(landmarks, width, height, is_right_handed)
        if direction is not None:
            source = "hand_grip_estimate"
            self.missing = 0
            target = math.atan2(direction[1], direction[0])
            response_seconds = 0.025
        else:
            self.missing += 1
            if self.angle is not None and self.missing <= max(1, round(self.fps*0.12)):
                return (math.cos(self.angle), math.sin(self.angle)), "held_grip_estimate"
            wi, ei = (16, 14) if is_right_handed else (15, 13)
            wrist, elbow = landmarks[wi], landmarks[ei]
            dx, dy = (wrist.x-elbow.x)*width, (wrist.y-elbow.y)*height
            if not math.isfinite(math.hypot(dx, dy)) or math.hypot(dx, dy) < 1e-6:
                if self.angle is None:
                    return None, "unavailable"
                return (math.cos(self.angle), math.sin(self.angle)), "held_grip_estimate"
            target = math.atan2(dy, dx)
            source = "forearm_fallback"
            response_seconds = 0.08
        if self.angle is None:
            self.angle = target
        else:
            delta = math.atan2(math.sin(target-self.angle), math.cos(target-self.angle))
            alpha = 1.0 - math.exp(-1.0/(self.fps*response_seconds))
            self.angle += alpha * delta
        return (math.cos(self.angle), math.sin(self.angle)), source
