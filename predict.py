import numpy as np, cv2, tensorflow as tf, os
IMAGE_SIZE  = 256
N_CHANNELS  = 5
CLIP_LIMIT  = 2.0
TILE_GRID   = 8
SOBEL_KSIZE = 3
DISC_PATH = os.environ.get("DISC_MODEL_PATH","disc.keras")
CUP_PATH = os.environ.get("CUP_MODEL_PATH", "cup.keras")

def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return bgr

def _build_stack(bgr) -> np.ndarray:
    rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe      = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID, TILE_GRID))
    clahe_gray = clahe.apply(gray)

    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = np.sqrt(gx * gx + gy * gy)
    maxv = float(mag.max())
    sobel_mag = (mag / maxv * 255.0 if maxv > 0 else mag).clip(0, 255).astype(np.uint8)

    stack = np.dstack([rgb, clahe_gray[..., None], sobel_mag[..., None]]).astype(np.float32)

    resized = np.stack(
        [cv2.resize(stack[..., c], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
         for c in range(N_CHANNELS)],
        axis=-1,
    )
    return (resized / 127.5 - 1.0)[np.newaxis]  # (1, 256, 256, 5)

def _predict_mask(model, stack: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a binary (H, W) uint8 mask from a (1, H, W, 5) stack."""
    prob = model.predict(stack, verbose=0)   # (1, 256, 256, 1)
    return (prob[0, ..., 0] > threshold).astype(np.uint8)

def _fit_ellipse_params(mask: np.ndarray):
    """
    Fit an ellipse to the largest contour in a binary mask.
    Returns (cx, cy, a, b, ang_rad) or None if fitting fails.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
    a, b = MA * 0.5, ma * 0.5
    return cx, cy, a, b, np.deg2rad(ang)

def _vertical_cdr(disc: np.ndarray, cup: np.ndarray) -> float | None:
    """
    Compute the vertical cup-to-disc ratio from ellipses fitted to the
    disc and cup masks.

    More robust than pixel-based CDR for noisy or irregular prediction masks,
    because the ellipse acts as a smooth regulariser over the raw segmentation.

    Returns the vertical CDR as a float, or None if either ellipse fit fails
    or the disc has zero vertical extent.
    """
    d_p = _fit_ellipse_params(disc)
    c_p = _fit_ellipse_params(cup)

    if d_p is None or c_p is None:
        return None
    
    _, _, a_d, b_d, ang_d = d_p
    _, _, a_c, b_c, ang_c = c_p


    # Vertical CDR: vertical half-extent of each ellipse's bounding box
    disc_half_h = np.sqrt((a_d * np.sin(ang_d)) ** 2 + (b_d * np.cos(ang_d)) ** 2)
    cup_half_h  = np.sqrt((a_c * np.sin(ang_c)) ** 2 + (b_c * np.cos(ang_c)) ** 2)
    vertical_cdr = float(cup_half_h / disc_half_h) if disc_half_h > 0 else None

    return vertical_cdr

def _encode_png(bgr)-> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("failed to encode overlay")
    return buf.tobytes()

def _draw_ellipse(img_bgr: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2):
    """Fit an ellipse to the largest contour in mask and draw it on img_bgr in-place."""
    params = _fit_ellipse_params(mask)
    if params is None:
        return
    cx, cy, a, b, ang_rad = params
    cv2.ellipse(img_bgr, (int(cx), int(cy)), (int(a), int(b)),
                np.degrees(ang_rad), 0, 360, color, thickness)


def _draw_overlay(bgr, disc_mask, cup_mask):
    RED    = (0, 0, 255)

    # Resize masks to match the display image before fitting ellipses
    h, w   = bgr.shape[:2]
    def _resize(m): return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    out = bgr.copy() #do not mutate the input since cv2.ellipse draws in-place
    _draw_ellipse(out, _resize(disc_mask), RED)
    _draw_ellipse(out, _resize(cup_mask), RED)
    
    return out



def load_models():
    d = tf.keras.models.load_model(DISC_PATH, compile=False)
    c = tf.keras.models.load_model(CUP_PATH, compile=False)
    return (d,c)

def predict(image_bytes: bytes, disc_model, cup_model) -> dict:
    bgr = _decode(image_bytes)
    stack = _build_stack(bgr)
    disc = _predict_mask(disc_model, stack)
    cup = _predict_mask(cup_model, stack)
    vcdr = _vertical_cdr(disc,cup)
    overlay = _draw_overlay(bgr, disc, cup)
    overlay = _encode_png(overlay)

    return {"vertical_cdr":vcdr, "overlay_png":overlay}












