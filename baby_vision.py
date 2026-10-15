#!/usr/bin/env python3
# ============================================================
# BabyVision module
# - wraps your face / hand / object / attention logic
# - exposes step() that returns structured vision_state
# ============================================================

import cv2
import time
import torch
import mediapipe as mp
import os, glob
from ultralytics import YOLOWorld

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "yolov8s-world.pt")
YOLO_MODEL_PATH = LOCAL_MODEL_PATH if os.path.isfile(LOCAL_MODEL_PATH) else "yolov8s-world.pt"
YOLO_SKIP = 3                          # Run YOLO every N frames
DWELL_TIME = 2.0                       # Seconds to dwell on target (not used yet but kept)
IOU_THRESHOLD = 0.05                   # IoU to say "object is in hand"


# ------------------------------------------------------------
# Utility: attention class (your same logic)
# ------------------------------------------------------------
class Attention:
    def __init__(self):
        self.target = None
        self.last_switch = 0.0

    def update(self, faces, hands, objects_in_hand):
        now = time.time()
        if objects_in_hand:
            # choose biggest object in hand
            obj = max(objects_in_hand, key=lambda o: o["bbox"][2] * o["bbox"][3])
            self.target = {"type": "object", **obj}
            self.last_switch = now
        elif faces:
            # choose biggest face
            face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            self.target = {"type": "face", **face}
            self.last_switch = now
        else:
            self.target = None
        return self.target


# ------------------------------------------------------------
# BabyVision class
# ------------------------------------------------------------
class BabyVision:
    def __init__(self, camera_index=0):
        # init MediaPipe face + hands
        mp_face = mp.solutions.face_detection
        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        mp_hands = mp.solutions.hands
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=6,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # init YOLO-World
        self.yolo_model = YOLOWorld(YOLO_MODEL_PATH)

        # device select
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # attention tracker
        self.attention = Attention()

        # camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                "Camera unavailable. Check macOS Camera permission for the app used to launch BabyX."
            )

        # state
        self.frame_idx = 0
        self.last_objects = []  # cache objects between YOLO frames
        self.last_frame = None  # 👈 NEW: GUI can grab most recent frame for saving crops

        # save dir for crops
        self.SAVE_DIR = os.path.join(BASE_DIR, "cropped_objects")
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _get_faces(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_detector.process(rgb)
        faces = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x, y = int(bb.xmin * w), int(bb.ymin * h)
                ww, hh = int(bb.width * w), int(bb.height * h)
                faces.append({
                    "bbox": (x, y, ww, hh),
                    "conf": float(d.score[0]),
                })
        return faces

    def _get_hands(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hand_detector.process(rgb)
        hands = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                xs = [pt.x for pt in lm.landmark]
                ys = [pt.y for pt in lm.landmark]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                hands.append({
                    "hand": handed.classification[0].label.lower(),
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "conf": float(handed.classification[0].score),
                })
        return hands

    def _get_objects(self, frame, conf=0.35, max_k=20):
        results = self.yolo_model.predict(
            source=frame,
            imgsz=640,
            conf=conf,
            max_det=max_k,
            device=self.device,
            verbose=False,
        )
        out = []
        if results:
            r = results[0]
            for b in r.boxes:
                det_conf = float(b.conf[0])
                cls = int(b.cls[0])
                label = r.names[cls]

                # skip people
                if label.lower() == "person":
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                out.append({
                    "label": label,
                    "conf": det_conf,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                })
        return out

    def _iou(self, boxA, boxB):
        (xA, yA, wA, hA) = boxA
        (xB, yB, wB, hB) = boxB
        x1, y1 = max(xA, xB), max(yA, yB)
        x2, y2 = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (wA * hA) + (wB * hB) - inter
        return inter / union if union > 0 else 0

    def _compute_face_side(self, frame_w, face_bbox):
        """Return 'left' / 'center' / 'right' for biggest face bbox."""
        if face_bbox is None:
            return None
        (x, y, w, h) = face_bbox
        face_center_x = x + w / 2.0

        left_region = frame_w / 3.0
        right_region = 2.0 * frame_w / 3.0

        if face_center_x < left_region:
            return "left"
        elif face_center_x > right_region:
            return "right"
        else:
            return "center"

    def save_cropped_object(self, frame, bbox, label="object"):
        """
        You already had this logic. GUI will call this after Teach.
        We leave it the same so nothing breaks.
        """
        (x, y, w, h) = bbox
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            print("⚠️ Skipped saving empty crop.")
            return None

        existing = sorted(glob.glob(os.path.join(self.SAVE_DIR, "cropped_*.jpg")))
        if existing:
            last_num = int(os.path.basename(existing[-1]).split("_")[1].split(".")[0])
            next_num = last_num + 1
        else:
            next_num = 0

        filename = os.path.join(self.SAVE_DIR, f"cropped_{next_num:03d}.jpg")
        cv2.imwrite(filename, crop)
        print(f"💾 Saved {label} → {filename}")
        return filename, crop  # GUI can also send this crop to object_store later

    # ----------------------------
    # PUBLIC STEP
    # ----------------------------
    def step(self):
        """
        Reads one frame from camera,
        runs detection / attention,
        returns (vision_state, frame) where:

        vision_state = {
            "focus_type":   "object"/"face"/"none",
            "focus_label":  e.g. "phone",
            "focus_hand":   e.g. "left"/"right"/None,
            "focus_side":   e.g. "left"/"center"/"right"/None,
            "current_bbox": (x,y,w,h) for attended object/face or None,
            "current_crop": cropped image (numpy array) or None,
            "confidence":   float 0.0-1.0 for what we're attending
        }

        frame = current BGR frame (mirrored)
        """

        ok, frame = self.cap.read()
        if not ok:
            return None, None

        # 👇 mirror so it feels like selfie cam
        frame = cv2.flip(frame, 1)

        # save last frame for GUI / Teach
        self.last_frame = frame.copy()

        self.frame_idx += 1

        faces = self._get_faces(frame)
        hands = self._get_hands(frame)

        # run YOLO every YOLO_SKIP frames, else reuse last_objects
        if self.frame_idx % YOLO_SKIP == 0:
            objects = self._get_objects(frame)
            self.last_objects = objects
        else:
            objects = self.last_objects

        # link objects to hands (object in hand)
                # link objects to hands (object in hand)
        objects_in_hand = []
        for obj in objects:
            for hand in hands:
                if self._iou(obj["bbox"], hand["bbox"]) > IOU_THRESHOLD:
                    objects_in_hand.append({        # <-- fixed name
                        **obj,
                        "hand": hand["hand"],
                    })

        # pick attention target
        target = self.attention.update(faces, hands, objects_in_hand)

        # build fields for vision_state
        focus_type = "none"
        focus_label = None
        focus_hand = None
        focus_side = None
        current_bbox = None
        current_crop = None
        confidence = 0.0  # 👈 NEW

        if target:
            focus_type = target["type"]  # "object" or "face"
            current_bbox = target["bbox"]

            if focus_type == "object":
                focus_label = target.get("label")
                focus_hand = target.get("hand")
                confidence = float(target.get("conf", 0.0))
            elif focus_type == "face":
                focus_label = "face"
                confidence = float(target.get("conf", 0.0))

        # compute face side for gaze (biggest face)
        frame_h, frame_w = frame.shape[:2]
        biggest_face_bbox = None
        if faces:
            biggest_face_bbox = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])["bbox"]
        focus_side = self._compute_face_side(frame_w, biggest_face_bbox)

        # crop attended bbox so Teach can save it easily
        if current_bbox is not None:
            (bx, by, bw, bh) = current_bbox
            current_crop = frame[by:by+bh, bx:bx+bw]

        vision_state = {
            "focus_type": focus_type,
            "focus_label": focus_label,
            "focus_hand": focus_hand,
            "focus_side": focus_side,
            "current_bbox": current_bbox,
            "current_crop": current_crop,
            "confidence": confidence,   # 👈 NEW so GUI memory_text works
        }

        return vision_state, frame

    # ----------------------------
    # Cleanup camera when we're done
    # ----------------------------
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


# ------------------------------------------------------------
# OPTIONAL MANUAL TEST
# (you can run: python baby_vision.py
#  press q to quit)
# ------------------------------------------------------------
if __name__ == "__main__":
    bv = BabyVision()
    try:
        while True:
            vision_state, frame = bv.step()
            if vision_state is None:
                break

            print(vision_state)

            # debug preview
            preview = frame.copy()
            if vision_state["current_bbox"] is not None:
                (x, y, w, h) = vision_state["current_bbox"]
                cv2.rectangle(preview, (x, y), (x+w, y+h), (0,255,255), 2)

            cv2.imshow("BabyVision debug", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        bv.release()
