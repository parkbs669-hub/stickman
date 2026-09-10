"""Humanized entry point for the Tennis Stickman renderer.

The validated v12.4 pipeline stays intact; only the drawing stage is replaced by
stickman_visual_upgrade so motion, cache, audio and impact handling remain shared.
"""
import sys
import tennis_stickman_v12_4_upgraded as base
from stickman_visual_upgrade import install

install(base)

if __name__ == "__main__":
    sys.exit(base.main())
