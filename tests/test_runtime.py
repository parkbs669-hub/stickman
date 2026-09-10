"""Focused runtime regressions; real FFmpeg cases use tiny synthetic videos."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stickman_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.directory = Path(self.temp.name)
        self.options = dict(name="포핸드", speed=1.0, strobe_frames=32, strobe_step=4,
                            height=720, ssaa=2, start=0.0, duration=None, min_impact_gap=0.2)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_options_and_extreme_tempos(self):
        runtime.validate_options(**self.options)
        for speed in (0.05, 0.125, 0.3, 0.5, 1, 1.5, 2, 4, 8):
            factors = [float(token.split("=")[1]) for token in runtime.build_atempo_filter(speed).split(",")]
            self.assertTrue(all(0.5 <= value <= 2 for value in factors))
            self.assertAlmostEqual(math.prod(factors), speed)

    def test_bad_options_are_rejected(self):
        invalid = {"name": ["", " ", "../escape", "C:\\name", "NUL.mp4", "COM1", "ends.", "bad|name"],
                   "speed": [0, -1, float("inf"), float("nan"), 0.049, 8.1],
                   "strobe_frames": [0, -1, 1.5], "strobe_step": [0, True],
                   "height": [239, 241, 2162], "ssaa": [0, 4, 1.5],
                   "start": [-0.1, float("nan")], "duration": [0, -1], "min_impact_gap": [0, -1]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    runtime.validate_options(**(self.options | {key: value}))

    def test_identity_hashes_entire_file(self):
        path = self.directory / "source.bin"
        data = b"a" * (2 * 1024 * 1024) + b"first"
        path.write_bytes(data)
        first = runtime.video_identity(path)
        self.assertEqual(first, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        with path.open("r+b") as stream:
            stream.seek(-5, 2)
            stream.write(b"other")
        second = runtime.video_identity(path)
        self.assertNotEqual(first, second)
        self.assertEqual(first["size"], second["size"])

    def test_cache_roundtrip_and_settings_invalidation(self):
        path = self.directory / "cache.json"
        metadata = {"identity": {"sha256": "test", "size": 4}, "fps": 30.0, "start": 2.0}
        frames = [None, [SimpleNamespace(x=i/33, y=0.4, z=-0.2, visibility=0.9) for i in range(33)]]
        runtime.save_pose_cache(path, metadata, frames)
        restored = runtime.load_pose_cache(path, metadata)
        self.assertIsNone(restored[0])
        self.assertEqual(restored[1][32].x, 32/33)
        self.assertEqual(restored[1][0].visibility, 0.9)
        self.assertIsNone(runtime.load_pose_cache(path, metadata | {"start": 1.0}))
        self.assertIsNone(runtime.load_pose_cache(path, metadata | {"identity": {"sha256": "changed", "size": 4}}))

    def test_cache_rejects_malformed_or_nonfinite_landmarks(self):
        path = self.directory / "bad.json"
        frames = [[[0, 1, 2, 0.9] for _ in range(33)]]
        base = {"schema_version": 1, "metadata": {}, "frames": frames}
        for bad in ([], [frames[0][:-1]], [[[0, 1, 2]] * 33], [[[0, 1, float("nan"), 1]] * 33],
                    [[[0, 1, True, 1]] * 33]):
            # Empty is a valid zero-frame cache; root checks extraction length.
            if bad == []:
                continue
            path.write_text(json.dumps(base | {"frames": bad}), encoding="utf-8")
            self.assertIsNone(runtime.load_pose_cache(path, {}))
        path.write_text("{broken", encoding="utf-8")
        self.assertIsNone(runtime.load_pose_cache(path, {}))

    def test_atomic_failure_preserves_existing_file(self):
        path = self.directory / "cache.json"
        runtime.atomic_write_json(path, {"ok": True})
        with self.assertRaises(ValueError):
            runtime.atomic_write_json(path, {"bad": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"ok": True})
        self.assertEqual(list(self.directory.iterdir()), [path])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and importlib.util.find_spec("numpy"),
                     "FFmpeg, FFprobe and numpy are required for integration tests")
class FFmpegIntegrationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def video(self, name, frames=20, fps=20):
        path = self.directory / name
        frame = self.np.zeros((48, 64, 3), dtype=self.np.uint8)
        with runtime.FFmpegVideoWriter(self.ffmpeg, path, 64, 48, fps) as writer:
            for index in range(frames):
                frame[:, :, 0] = index % 255
                writer.write(frame)
        return path

    def probe(self, path):
        return runtime._probe_json(self.ffprobe, path, "stream=codec_type,duration,nb_frames:format=duration")

    def test_writer_produces_expected_duration_and_frame_count(self):
        path = self.video("writer.mp4", 30, 15)
        stream = self.probe(path)["streams"][0]
        self.assertEqual(int(stream["nb_frames"]), 30)
        self.assertAlmostEqual(float(stream["duration"]), 2, places=4)

    def test_failed_encoder_and_exception_close_resources(self):
        writer = None
        with self.assertRaises(RuntimeError):
            with runtime.FFmpegVideoWriter(self.ffmpeg, self.directory / "failed.mp4", 64, 48, 20,
                                           preset="definitely_invalid") as writer:
                writer.write(self.np.zeros((48, 64, 3), dtype=self.np.uint8))
        self.assertIsNotNone(writer._process.poll())
        self.assertTrue(writer._stderr.closed)
        with self.assertRaisesRegex(RuntimeError, "user error"):
            with runtime.FFmpegVideoWriter(self.ffmpeg, self.directory / "abort.mp4", 64, 48, 20) as writer:
                raise RuntimeError("user error")
        self.assertIsNotNone(writer._process.poll())
        self.assertTrue(writer._stderr.closed)

    def test_silent_source_is_copied_without_loss(self):
        source = self.video("silent.mp4")
        destination = self.directory / "copied.mp4"
        runtime.mux_audio(self.ffmpeg, self.ffprobe, source, source, destination)
        self.assertEqual(source.read_bytes(), destination.read_bytes())

    def test_short_audio_does_not_truncate_video_at_fast_speed(self):
        video = self.video("video.mp4", frames=40, fps=20)
        source = self.directory / "short.wav"
        runtime._run_checked([self.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                              "sine=frequency=440:duration=0.3", source], "fixture")
        destination = self.directory / "padded.mp4"
        runtime.mux_audio(self.ffmpeg, self.ffprobe, video, source, destination, speed=4)
        metadata = self.probe(destination)
        video_stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        audio_stream = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
        self.assertEqual(int(video_stream["nb_frames"]), 40)
        self.assertAlmostEqual(float(video_stream["duration"]), 2, places=3)
        self.assertAlmostEqual(float(audio_stream["duration"]), 2, delta=0.03)

    def test_audio_trim_uses_source_time_and_tempo_preserves_pitch(self):
        rate = 16000
        first = self.np.sin(2 * self.np.pi * 440 * self.np.arange(rate) / rate)
        second = self.np.sin(2 * self.np.pi * 880 * self.np.arange(rate) / rate)
        source = self.directory / "two_tones.wav"
        with wave.open(str(source), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes((self.np.concatenate([first, second]) * 10000).astype("<i2").tobytes())
        video = self.video("slowed.mp4", frames=40, fps=20)
        destination = self.directory / "trimmed.mp4"
        runtime.mux_audio(self.ffmpeg, self.ffprobe, video, source, destination,
                          speed=0.25, start=1.0, duration=2.0)
        samples = self.np.frombuffer(runtime._run_checked(
            [self.ffmpeg, "-v", "error", "-i", destination, "-map", "0:a:0", "-f", "f32le",
             "-ac", "1", "-ar", str(rate), "pipe:1"], "decode"), dtype="<f4")
        window = samples[rate // 2:rate]
        spectrum = abs(self.np.fft.rfft(window))
        peak_hz = self.np.fft.rfftfreq(len(window), 1 / rate)[spectrum.argmax()]
        self.assertAlmostEqual(float(peak_hz), 880, delta=6)
        self.assertAlmostEqual(float(self.probe(destination)["format"]["duration"]), 2, delta=0.03)

    def test_bad_source_surfaces_probe_error(self):
        video = self.video("valid.mp4")
        source = self.directory / "bad.mp4"
        source.write_bytes(b"not a media file")
        with self.assertRaisesRegex(RuntimeError, "FFprobe failed"):
            runtime.mux_audio(self.ffmpeg, self.ffprobe, video, source, self.directory / "result.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
