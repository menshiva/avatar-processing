import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class FaceDetector:
    detector: vision.FaceDetector
    lock: threading.Lock


def initialize(model_path: str, conf: float) -> FaceDetector:
    return FaceDetector(
        vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=conf,
        )),
        threading.Lock()
    )


def detect_get_crop_view(face_detector: FaceDetector, img: mp.Image, margin_percent: float) -> npt.NDArray[np.uint8] | None:
    detector_result: vision.FaceDetectorResult | None = None
    with face_detector.lock:
        detector_result = face_detector.detector.detect(img)

    if detector_result is None or len(detector_result.detections) != 1:
        return None

    image_view = img.numpy_view()
    h, w = image_view.shape[:2]
    bbox = detector_result.detections[0].bounding_box

    dx = int(bbox.width * margin_percent)
    dy = int(bbox.height * margin_percent)

    # Clamp to image bounds
    x = max(0, bbox.origin_x - dx)
    y = max(0, bbox.origin_y - 3 * dy)
    x2 = min(w, x + bbox.width + dx * 2)
    y2 = min(h, y + bbox.height + dy * 4)
    if x2 <= x or y2 <= y:
        return None

    return image_view[y:y2, x:x2]
