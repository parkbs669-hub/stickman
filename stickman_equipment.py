"""Body-scaled racket geometry independent of pose orientation and head width."""
import math
import statistics

RACKET_TO_BODY_HEIGHT = 0.38


def body_height_pixels(frames, width, height, scale=.9):
    estimates = []
    for frame in frames:
        if frame is None or len(frame) < 33:
            continue
        def point(index):
            return frame[index].x*width*scale, frame[index].y*height*scale
        def distance(a,b):
            return math.dist(point(a),point(b))
        shoulder = tuple((a+b)/2 for a,b in zip(point(11),point(12)))
        hip = tuple((a+b)/2 for a,b in zip(point(23),point(24)))
        torso = math.dist(shoulder,hip)
        legs = (distance(23,25)+distance(25,27)+distance(24,26)+distance(26,28))/2
        head_radius = max(distance(11,12)*.55, height*(16/720))
        estimate = torso + legs + head_radius*2.8
        if math.isfinite(estimate) and estimate > height*.15:
            estimates.append(estimate)
    return statistics.median(estimates) if estimates else height*.7


def racket_dimensions(length, face_ratio=1.0):
    """Adult-style proportions: ~half shaft/throat and half hoop, narrower head."""
    length=max(8.0,float(length))
    face_ratio=max(.12,min(1.0,float(face_ratio)))
    return {"length":length, "grip_end":length*.24,
            "hoop_base":length*.48, "center":length*.73,
            "head_long_radius":length*.25,
            "head_short_radius":length*.175*face_ratio}
