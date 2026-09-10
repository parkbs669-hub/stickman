"""Estimate racket shaft direction from the grip, rather than extending the arm.

MediaPipe Pose's index/pinky knuckle axis is only an estimate of the grip axis.
Collapsed or occluded finger landmarks must not be treated as reliable angles.
"""
import math


HAND_CONFIDENCE = 0.5
FOREARM_CONFIDENCE = 0.35
GRIP_HOLD_SECONDS = 0.12


def _confidence(point):
    """Return a finite landmark confidence; malformed values are treated as missing."""
    try:
        value = float(getattr(point, "v", getattr(point, "visibility", 1.0)))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _screen_delta(a, b, width, height):
    """Return a finite pixel-space vector from b to a, otherwise None."""
    try:
        width = float(width)
        height = float(height)
        dx = (float(a.x) - float(b.x)) * width
        dy = (float(a.y) - float(b.y)) * height
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (width, height, dx, dy)) or width <= 0 or height <= 0:
        return None
    return dx, dy


def grip_direction(landmarks, width, height, is_right_handed=True):
    """Return a directed screen-space unit vector, or None for ambiguous fingers."""
    if landmarks is None or len(landmarks) < 23:
        return None
    pinky, index, wrist, elbow = (18, 20, 16, 14) if is_right_handed else (17, 19, 15, 13)
    p, i, w, e = (landmarks[j] for j in (pinky, index, wrist, elbow))
    if min(_confidence(p), _confidence(i), _confidence(w)) < HAND_CONFIDENCE:
        return None

    # Work in pixel space: normalized x/y have different scales in portrait video.
    grip = _screen_delta(i, p, width, height)
    if grip is None:
        return None
    dx, dy = grip
    length = math.hypot(dx, dy)

    # The forearm length is only a scale reference. If its coordinates are malformed,
    # keep the absolute/image-size thresholds rather than trusting a NaN scale.
    arm_delta = _screen_delta(w, e, width, height)
    arm = math.hypot(*arm_delta) if arm_delta is not None else 0.0
    min_length = max(2.5, float(height) * 0.005, arm * 0.07)
    if not math.isfinite(length) or length < min_length:
        return None
    return dx / length, dy / length


def _forearm_direction(landmarks, width, height, is_right_handed):
    """Return a reliable wrist-from-elbow direction for fallback, otherwise None."""
    if landmarks is None or len(landmarks) < 17:
        return None
    wi, ei = (16, 14) if is_right_handed else (15, 13)
    wrist, elbow = landmarks[wi], landmarks[ei]
    if min(_confidence(wrist), _confidence(elbow)) < FOREARM_CONFIDENCE:
        return None
    delta = _screen_delta(wrist, elbow, width, height)
    if delta is None:
        return None
    length = math.hypot(*delta)
    if not math.isfinite(length) or length < 1e-6:
        return None
    return delta[0] / length, delta[1] / length


class RacketDirectionTracker:
    def __init__(self, fps):
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be a finite positive number")
        self.angle = None
        self.missing = 0

    def reset(self):
        self.angle = None
        self.missing = 0

    def _smooth(self, target, response_seconds):
        if self.angle is None:
            self.angle = target
        else:
            delta = math.atan2(math.sin(target - self.angle), math.cos(target - self.angle))
            alpha = 1.0 - math.exp(-1.0 / (self.fps * response_seconds))
            self.angle += alpha * delta
            # Keep the stored state bounded so long renders cannot accumulate turns.
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
        return math.cos(self.angle), math.sin(self.angle)

    def update(self, landmarks, width, height, is_right_handed=True):
        if landmarks is None:
            self.reset()
            return None, "missing_pose"

        direction = grip_direction(landmarks, width, height, is_right_handed)
        if direction is not None:
            self.missing = 0
            target = math.atan2(direction[1], direction[0])
            return self._smooth(target, 0.025), "hand_grip_estimate"

        self.missing += 1
        hold_frames = max(1, round(self.fps * GRIP_HOLD_SECONDS))
        if self.angle is not None and self.missing <= hold_frames:
            return (math.cos(self.angle), math.sin(self.angle)), "held_grip_estimate"

        direction = _forearm_direction(landmarks, width, height, is_right_handed)
        if direction is None:
            # Do not freeze a stale racket indefinitely through a long occlusion.
            self.angle = None
            return None, "unavailable"

        target = math.atan2(direction[1], direction[0])
        return self._smooth(target, 0.08), "forearm_fallback"
