"""Example detector stub for eval.py. Replace with YOUR detector.
Contract: print one number in [0,1] = probability the watermark / AI-origin is
still present in the image at argv[1]. eval.py reads the last whitespace token.

This stub decodes an invisible-watermark bit string and returns bit-accuracy,
so you can smoke-test the loop before plugging in your real detector:
    python eval.py --images my_wm_images --detector "python detector_example.py {img}"
"""
import sys
import numpy as np
from PIL import Image
try:
    import cv2
    from imwatermark import WatermarkDecoder
    img = cv2.imread(sys.argv[1])
    bits = WatermarkDecoder("bytes", 32).decode(img, "dwtDct")
    ref = np.unpackbits(np.frombuffer(b"test", dtype=np.uint8))
    got = np.unpackbits(np.frombuffer(bytes(bits), dtype=np.uint8))
    print(float((ref == got).mean()))
except Exception:
    print(0.0)
