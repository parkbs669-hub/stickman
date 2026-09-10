from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import copy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stickman_equipment import body_height_pixels, racket_dimensions, RACKET_TO_BODY_HEIGHT


def person():
    p=[SimpleNamespace(x=.5,y=.5,z=0,v=1) for _ in range(33)]
    for i,xy in {11:(.4,.25),12:(.6,.25),23:(.44,.50),24:(.56,.50),
                 25:(.4,.7),26:(.6,.7),27:(.4,.9),28:(.6,.9)}.items():p[i].x,p[i].y=xy
    return p


class EquipmentTests(unittest.TestCase):
    def test_body_scale_uses_robust_clip_median(self):
        normal=person(); bad=copy.deepcopy(normal)
        for lm in bad:lm.x*=20;lm.y*=20
        expected=body_height_pixels([normal]*10,400,720)
        self.assertEqual(body_height_pixels([normal]*10+[bad],400,720),expected)

    def test_equipment_scales_with_render_resolution(self):
        p=person()
        self.assertAlmostEqual(body_height_pixels([p],800,1440),2*body_height_pixels([p],400,720))

    def test_hoop_and_handle_human_style_proportions(self):
        dims=racket_dimensions(160)
        self.assertAlmostEqual(dims['center']+dims['head_long_radius'],156.8)
        self.assertAlmostEqual(dims['head_long_radius']*2/160,.5)
        self.assertAlmostEqual(dims['head_short_radius']*2/160,.35)
        self.assertGreater(dims['hoop_base'],dims['grip_end'])
        self.assertAlmostEqual(RACKET_TO_BODY_HEIGHT,.38)

    def test_face_turn_does_not_shorten_racket(self):
        a,b=racket_dimensions(160,1),racket_dimensions(160,.2)
        self.assertEqual(a['length'],b['length'])
        self.assertEqual(a['center'],b['center'])
        self.assertEqual(a['head_long_radius'],b['head_long_radius'])
        self.assertLess(b['head_short_radius'],a['head_short_radius'])


if __name__=='__main__':unittest.main(verbosity=2)
