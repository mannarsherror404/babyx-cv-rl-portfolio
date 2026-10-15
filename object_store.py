#!/usr/bin/env python3
# object_store.py — instance-level visual memory for BabyX (safe, path-stable)

import os
import cv2
import uuid
import pickle
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ---------- PATHS (absolute, relative to this file) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBJECT_MEMORY_FILE = os.path.join(BASE_DIR, "object_memory.pkl")
CROPS_DIR = os.path.join(BASE_DIR, "object_crops")
os.makedirs(CROPS_DIR, exist_ok=True)

# ---------- IO ----------
def _load_store() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(OBJECT_MEMORY_FILE):
        with open(OBJECT_MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def _save_store(store: Dict[str, Dict[str, Any]]) -> None:
    with open(OBJECT_MEMORY_FILE, "wb") as f:
        # protocol 4+ keeps files smaller/faster on py3
        pickle.dump(store, f, protocol=4)

# ---------- FEATURES ----------
def _extract_features_bgr(bgr_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    bgr_crop: np.array [H,W,3] (OpenCV BGR)
    returns: ORB descriptor matrix (N x 32) or None
    """
    if bgr_crop is None or not isinstance(bgr_crop, np.ndarray):
        return None
    h, w = bgr_crop.shape[:2]
    if h == 0 or w == 0:
        return None

    # Optional: lightly bound size for stable features/speed
    max_side = 512
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        bgr_crop = cv2.resize(bgr_crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    # Cap features for consistency/speed
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=7)
    kps, desc = orb.detectAndCompute(gray, None)
    return desc  # may be None

# ---------- PUBLIC API ----------
def remember_object(class_label: str, taught_name: str, bgr_crop: np.ndarray) -> Optional[str]:
    """
    Save a new object instance:
    - class_label like "cell phone"
    - taught_name like "Keshav's phone"
    - bgr_crop is the crop of that object from the camera frame (BGR)

    returns short object_id or None if crop invalid
    """
    # Guard bad crops
    if bgr_crop is None or bgr_crop.size == 0:
        print("[object_store] skip: empty crop")
        return None

    store = _load_store()
    desc = _extract_features_bgr(bgr_crop)

    obj_id = str(uuid.uuid4())[:8]
    crop_path = os.path.join(CROPS_DIR, f"{obj_id}.png")
    try:
        cv2.imwrite(crop_path, bgr_crop)
    except Exception as e:
        print("[object_store] warning: failed to write crop:", e)

    store[obj_id] = {
        "label": class_label,
        "name": taught_name,
        "descriptors": desc,      # may be None if low-texture
        "crop_path": crop_path,
    }

    _save_store(store)
    print(f"[object_store] saved {obj_id} as '{taught_name}' ({class_label})")
    return obj_id

def identify_object(bgr_crop: np.ndarray, min_matches: int = 20) -> Tuple[Optional[str], Optional[str], int]:
    """
    Given a new crop, try to match to known instances.
    returns (best_name, best_label, score) OR (None, None, 0)
    """
    store = _load_store()
    if not store:
        return (None, None, 0)

    query_desc = _extract_features_bgr(bgr_crop)
    if query_desc is None:
        return (None, None, 0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_score = 0
    best_name = None
    best_label = None

    for obj_id, obj in store.items():
        saved_desc = obj.get("descriptors")
        if saved_desc is None:
            continue
        matches = bf.match(query_desc, saved_desc)
        score = len(matches)
        if score > best_score:
            best_score = score
            best_name = obj.get("name")
            best_label = obj.get("label")

    # Apply a light threshold so random clutter doesn't bind
    if best_score < min_matches:
        return (None, None, 0)

    return (best_name, best_label, best_score)

# ---------- utilities (optional) ----------
def stats() -> Dict[str, int]:
    """Return counts per class label (for debugging UI)."""
    s = _load_store()
    out: Dict[str, int] = {}
    for v in s.values():
        out[v["label"]] = out.get(v["label"], 0) + 1
    return out

def clear_store() -> None:
    """Danger: clears all instance memory (doesn't delete crops on disk)."""
    _save_store({})
    print("[object_store] cleared object_memory.pkl")