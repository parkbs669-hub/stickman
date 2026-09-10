"""Validated I/O helpers for the tennis stickman generator (standard library only)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace


CACHE_SCHEMA_VERSION = 2


def _finite_number(value, label, *, minimum=None, maximum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if strictly_positive and value <= 0:
        raise ValueError(f"{label} must be greater than zero")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")


def validate_options(name, speed, strobe_frames, strobe_step, height, ssaa,
                     start=0.0, duration=None, min_impact_gap=0.20):
    """Reject unsafe output names and invalid render options before doing work."""
    if not isinstance(name, str) or not name.strip() or name in (".", ".."):
        raise ValueError("name must be a non-empty output basename")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((" ", ".")):
        raise ValueError("name contains characters that cannot be used in a Windows filename")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "¹²³")
    if name.split(".", 1)[0].upper() in reserved:
        raise ValueError("name is a reserved Windows filename")
    try:
        name_units = len(name.encode("utf-16-le")) // 2
    except UnicodeEncodeError as exc:
        raise ValueError("name contains invalid Unicode") from exc
    if name_units > 180:
        raise ValueError("name must be 180 Windows filename characters or fewer")
    _finite_number(speed, "speed", minimum=0.05, maximum=8.0)
    for value, label in ((strobe_frames, "strobe_frames"), (strobe_step, "strobe_step")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or not 240 <= height <= 2160 or height % 2:
        raise ValueError("height must be an even integer between 240 and 2160")
    if isinstance(ssaa, bool) or not isinstance(ssaa, int) or not 1 <= ssaa <= 3:
        raise ValueError("ssaa must be an integer between 1 and 3")
    _finite_number(start, "start", minimum=0.0)
    if duration is not None:
        _finite_number(duration, "duration", strictly_positive=True)
    _finite_number(min_impact_gap, "min_impact_gap", strictly_positive=True)


def build_atempo_filter(speed):
    """Return tempo factors within FFmpeg's conservative 0.5..2.0 range."""
    _finite_number(speed, "speed", minimum=0.05, maximum=8.0)
    remaining = float(speed)
    factors = []
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={factor:.12g}" for factor in factors)


def video_identity(path):
    """Hash the entire source in bounded memory, detecting concurrent changes."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    if size != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while computing its identity: {path}")
    return {"sha256": digest.hexdigest(), "size": size}


def atomic_write_json(path, data):
    """Replace a JSON document only after its new contents are flushed to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _encode_frames(frames):
    if not isinstance(frames, (list, tuple)):
        raise ValueError("Cached frames must be a list")
    encoded = []
    for frame in frames:
        if frame is None:
            encoded.append(None)
            continue
        if len(frame) != 33:
            raise ValueError("Each detected pose must contain exactly 33 landmarks")
        landmarks = []
        for landmark in frame:
            values = [float(landmark.x), float(landmark.y), float(landmark.z),
                      float(getattr(landmark, "v", getattr(landmark, "visibility", 1.0)))]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Pose landmarks must contain only finite numbers")
            landmarks.append(values)
        encoded.append(landmarks)
    return encoded


def save_pose_cache(path, metadata, frames):
    """Store validated raw pose data; metadata must describe every extraction setting."""
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata must be a dictionary")
    atomic_write_json(path, {"schema_version": CACHE_SCHEMA_VERSION,
                             "metadata": metadata, "frames": _encode_frames(frames)})


def load_pose_cache(path, metadata):
    """Return landmarks only for an exact metadata match, otherwise a cache miss."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != CACHE_SCHEMA_VERSION):
            return None
        # Canonical JSON distinguishes values such as true and 1, unlike Python equality.
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        if not isinstance(metadata, dict) or canonical(data.get("metadata")) != canonical(metadata):
            return None
        if not isinstance(data.get("frames"), list):
            return None
        frames = []
        for frame in data["frames"]:
            if frame is None:
                frames.append(None)
                continue
            if not isinstance(frame, list) or len(frame) != 33:
                return None
            landmarks = []
            for values in frame:
                if not isinstance(values, list) or len(values) != 4:
                    return None
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                    return None
                landmarks.append(SimpleNamespace(x=values[0], y=values[1], z=values[2], v=values[3], visibility=values[3]))
            frames.append(landmarks)
        expected = metadata.get("frame_count")
        if expected is not None and (type(expected) is not int or len(frames) != expected):
            return None
        return frames
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def _process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class FFmpegVideoWriter:
    """Write BGR uint8 frames to H.264, always checking FFmpeg's exit status.

    ``path`` should be a staging file owned by the caller. The caller can publish
    it atomically after all rendering and optional audio muxing have succeeded.
    """

    def __init__(self, ffmpeg, path, width, height, fps, crf=18, preset="fast"):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 or v % 2 for v in (width, height)):
            raise ValueError("Video width and height must be positive even integers")
        _finite_number(fps, "fps", strictly_positive=True)
        self.path = Path(path)
        self.width, self.height = width, height
        self.frame_count = 0
        self._closed = False
        self._process = None
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                   "-r", f"{float(fps):.12g}", "-i", "pipe:0", "-an",
                   "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                             stderr=self._stderr, **_process_options())
        except BaseException:
            self._stderr.close()
            self._closed = True
            raise

    def __enter__(self):
        return self

    def _error_details(self):
        self._stderr.flush()
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell() - 16000))
        return self._stderr.read().decode("utf-8", errors="replace").strip()

    def _abort(self):
        if self._closed:
            return
        process = self._process
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except (OSError, ValueError):
                        pass
        finally:
            self._closed = True
            self._stderr.close()

    def write(self, frame):
        if self._closed:
            raise RuntimeError("Video writer is already closed")
        if getattr(frame, "shape", None) != (self.height, self.width, 3) or str(getattr(frame, "dtype", "")) != "uint8":
            raise ValueError(f"Expected a uint8 BGR frame with shape {(self.height, self.width, 3)}")
        try:
            if self._process.poll() is not None:
                raise BrokenPipeError("FFmpeg exited before all frames were written")
            self._process.stdin.write(frame.tobytes(order="C"))
            self.frame_count += 1
        except (BrokenPipeError, OSError) as exc:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            details = self._error_details()
            self._abort()
            raise RuntimeError(f"FFmpeg failed while encoding video: {details or exc}") from exc

    def close(self):
        if self._closed:
            return
        process = self._process
        try:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                returncode = process.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("FFmpeg did not finish encoding within 120 seconds") from exc
            details = self._error_details()
            if returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed (exit {returncode}): {details}")
            if self.frame_count == 0 or not self.path.is_file() or self.path.stat().st_size == 0:
                raise RuntimeError("FFmpeg produced no video frames")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            self._closed = True
            self._stderr.close()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self._abort()
        else:
            self.close()
        return False

    def __del__(self):
        if getattr(self, "_closed", True) is False:
            try:
                self._abort()
            except Exception:
                pass


def _run_checked(command, label):
    result = subprocess.run([str(item) for item in command], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_process_options())
    if result.returncode:
        details = result.stderr.decode("utf-8", errors="replace")[-16000:].strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}): {details}")
    return result.stdout


def _probe_json(ffprobe, path, entries, selector=None):
    command = [ffprobe, "-v", "error"]
    if selector:
        command += ["-select_streams", selector]
    command += ["-show_entries", entries, "-of", "json", path]
    try:
        return json.loads(_run_checked(command, "FFprobe"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("FFprobe returned invalid metadata") from exc


def mux_audio(ffmpeg, ffprobe, video_path, source_path, destination,
              speed=1.0, start=0.0, duration=None):
    """Add source audio at source-time ``start`` with pitch-preserving tempo.

    ``duration`` is the final encoded video duration, not the source clip duration.
    Omit it to probe the encoded video. Audio is padded to preserve every video
    frame even when the source audio ends early; a source without audio is copied.
    """
    tempo = build_atempo_filter(speed)
    _finite_number(start, "start", minimum=0.0)
    video_path, source_path, destination = map(Path, (video_path, source_path, destination))
    if destination.resolve() in (video_path.resolve(), source_path.resolve()):
        raise ValueError("Audio destination must differ from both input files")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError("Encoded video is missing or empty")
    if duration is None:
        metadata = _probe_json(ffprobe, video_path, "stream=duration:format=duration", "v:0")
        candidates = [s.get("duration") for s in metadata.get("streams", [])]
        candidates.append(metadata.get("format", {}).get("duration"))
        for candidate in candidates:
            try:
                candidate = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
                break
        if duration is None:
            raise RuntimeError("Unable to determine encoded video duration")
    _finite_number(duration, "duration", strictly_positive=True)
    source_metadata = _probe_json(ffprobe, source_path, "stream=index", "a:0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source_metadata.get("streams"):
        shutil.copyfile(video_path, destination)
        return destination
    audio_filter = (f"[1:a:0]atrim=start={float(start):.12g},asetpts=PTS-STARTPTS,"
                    f"{tempo},apad,atrim=duration={float(duration):.12g}[audio]")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", video_path, "-i", source_path, "-filter_complex", audio_filter,
               "-map", "0:v:0", "-map", "[audio]", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-t", f"{float(duration):.12g}",
               "-movflags", "+faststart", destination]
    _run_checked(command, "Audio mux")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Audio mux produced no output")
    return destination
