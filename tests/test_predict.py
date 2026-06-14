import pytest
from predict import _vertical_cdr, _decode, _build_stack
import numpy as np
import cv2
H,W = 256, 256
N_CHANNELS = 5
c = (W//2,H//2)

def test_vertical_cdr_known_ratio():
    cup_mask, disc_mask = np.zeros((H,W),np.uint8), np.zeros((H,W),np.uint8)
    cv2.circle(img = cup_mask, center= c, radius = 40, color= 1,thickness=-1)
    cv2.circle(disc_mask, center= c, radius = 80, color= 1,thickness=-1)
    vcdr = _vertical_cdr(disc_mask,cup_mask)
    assert vcdr == pytest.approx(0.5, abs=0.05)

def test_vertical_cdr_empty_mask_returns_none():
    cup_mask, disc_mask = np.zeros((H,W),np.uint8), np.zeros((H,W),np.uint8)
    vcdr = _vertical_cdr(disc_mask,cup_mask)
    assert vcdr is None

def test_build_stack_shape_and_range():
    bgr_test = np.random.randint(low=0, high=256, size = (480,640,3),dtype=np.uint8) #deliberately choosing an image which needs resizing
    stack = _build_stack(bgr_test)
    assert stack.shape == (1, H, W, N_CHANNELS)
    assert stack.min() >= -1 and stack.max() <= 1.0

def test_decode_roundtrip():
    img = np.random.randint(low = 0, high = 256, size = (H,W,3),dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img) #img -> PNG bytes in memory
    png_bytes = buf.tobytes()
    recreated_img = _decode(png_bytes)
    assert np.array_equal(recreated_img, img)

def test_decode_rejects_garbage():
    with pytest.raises(ValueError): #this passes the test if the code within it raises a ValueError exception
        _decode(b"not an image")



