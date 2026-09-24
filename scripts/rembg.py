"""
Lightweight ONNX-native rembg shim that runs u2net directly via onnxruntime.
Avoids importing unsigned C-extensions from scipy/scikit-image blocked by Windows Application Control.
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

_session = None

def new_session(model_name="u2net"):
    model_path = os.path.expanduser(f"~/.u2net/{model_name}.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

def get_session():
    global _session
    if _session is None:
        _session = new_session("u2net")
    return _session

def remove(data, session=None, **kwargs):
    if not isinstance(data, Image.Image):
        im = Image.open(data)
    else:
        im = data

    orig_size = im.size  # (w, h)

    # Preprocess for u2net
    img_rgb = im.convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_arr = np.transpose(img_arr, (2, 0, 1))[np.newaxis, ...]

    sess = session or get_session()
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: img_arr})
    pred = out[0][0, 0]

    # Normalize to 0..255
    pred_min = float(np.min(pred))
    pred_max = float(np.max(pred))
    mask = (pred - pred_min) / (pred_max - pred_min + 1e-8)
    mask = (mask * 255.0).astype(np.uint8)

    mask_im = Image.fromarray(mask, mode="L").resize(orig_size, Image.Resampling.BILINEAR)

    result = im.convert("RGBA")
    result.putalpha(mask_im)
    return result
