import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import tennis_stickman_v12_4_upgraded as app


class TrackNetTests(unittest.TestCase):
    def test_bright_class_does_not_wrap_to_black(self):
        observed = []
        def circles(binary, *args, **kwargs):
            observed.append(binary.copy())
            return np.array([[[0., 2., 2.]]], dtype=np.float32)
        with patch.object(app.cv2, 'HoughCircles', side_effect=circles):
            x, y = app._tracknet_postprocess(np.full(32, 255, dtype=np.int64), w=8)
        self.assertTrue(np.all(observed[0] == 255))
        self.assertEqual(x, 0.0)

    def test_normalized_heatmap_supported(self):
        with patch.object(app.cv2, 'HoughCircles', return_value=None) as mock:
            self.assertEqual(app._tracknet_postprocess(np.ones(32, dtype=np.float32), w=8), (None, None))
            self.assertTrue(np.all(mock.call_args.args[0] == 255))

    @unittest.skipUnless(app._TORCH_AVAILABLE, 'Optional PyTorch absent')
    def test_model_output_preserves_pooling_dimensions(self):
        with app.torch.no_grad():
            model = app.BallTrackerNet().eval()
            result = model(app.torch.zeros(1, 9, 40, 64), testing=True)
        self.assertEqual(tuple(result.shape), (1, 256, 40*64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
