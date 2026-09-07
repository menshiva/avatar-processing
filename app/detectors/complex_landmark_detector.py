import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
import numpy as np
import numpy.typing as npt
import cv2
from dataclasses import dataclass
import threading


_LIPS_INNER_TESSELATION: list[FaceLandmarksConnections.Connection] = [
    FaceLandmarksConnections.Connection(78, 191),
    FaceLandmarksConnections.Connection(78, 95),
    FaceLandmarksConnections.Connection(95, 191),
    FaceLandmarksConnections.Connection(80, 191),
    FaceLandmarksConnections.Connection(80, 95),
    FaceLandmarksConnections.Connection(80, 81),
    FaceLandmarksConnections.Connection(81, 95),
    FaceLandmarksConnections.Connection(81, 88),
    FaceLandmarksConnections.Connection(88, 95),
    FaceLandmarksConnections.Connection(81, 178),
    FaceLandmarksConnections.Connection(88, 178),
    FaceLandmarksConnections.Connection(81, 82),
    FaceLandmarksConnections.Connection(82, 178),
    FaceLandmarksConnections.Connection(82, 87),
    FaceLandmarksConnections.Connection(87, 178),
    FaceLandmarksConnections.Connection(13, 82),
    FaceLandmarksConnections.Connection(13, 87),
    FaceLandmarksConnections.Connection(13, 14),
    FaceLandmarksConnections.Connection(14, 87),
    FaceLandmarksConnections.Connection(13, 312),
    FaceLandmarksConnections.Connection(14, 312),
    FaceLandmarksConnections.Connection(14, 317),
    FaceLandmarksConnections.Connection(312, 317),
    FaceLandmarksConnections.Connection(311, 312),
    FaceLandmarksConnections.Connection(311, 317),
    FaceLandmarksConnections.Connection(310, 311),
    FaceLandmarksConnections.Connection(310, 402),
    FaceLandmarksConnections.Connection(317, 402),
    FaceLandmarksConnections.Connection(311, 402),
    FaceLandmarksConnections.Connection(310, 318),
    FaceLandmarksConnections.Connection(318, 402),
    FaceLandmarksConnections.Connection(310, 415),
    FaceLandmarksConnections.Connection(324, 415),
    FaceLandmarksConnections.Connection(310, 324),
    FaceLandmarksConnections.Connection(318, 324),
    FaceLandmarksConnections.Connection(415, 308),
    FaceLandmarksConnections.Connection(308, 324),
]

_LEFT_EYE_TESSELATION: list[FaceLandmarksConnections.Connection] = [
    FaceLandmarksConnections.Connection(33, 246),
    FaceLandmarksConnections.Connection(246, 7),
    FaceLandmarksConnections.Connection(7, 33),
    FaceLandmarksConnections.Connection(246, 161),
    FaceLandmarksConnections.Connection(161, 7),
    FaceLandmarksConnections.Connection(161, 163),
    FaceLandmarksConnections.Connection(7, 163),
    FaceLandmarksConnections.Connection(161, 471),
    FaceLandmarksConnections.Connection(471, 163),
    FaceLandmarksConnections.Connection(161, 160),
    FaceLandmarksConnections.Connection(160, 471),
    FaceLandmarksConnections.Connection(471, 144),
    FaceLandmarksConnections.Connection(144, 163),
    FaceLandmarksConnections.Connection(160, 159),
    FaceLandmarksConnections.Connection(159, 471),
    FaceLandmarksConnections.Connection(471, 145),
    FaceLandmarksConnections.Connection(145, 144),
    FaceLandmarksConnections.Connection(159, 468),
    FaceLandmarksConnections.Connection(468, 471),
    FaceLandmarksConnections.Connection(468, 145),
    FaceLandmarksConnections.Connection(159, 158),
    FaceLandmarksConnections.Connection(158, 468),
    FaceLandmarksConnections.Connection(468, 469),
    FaceLandmarksConnections.Connection(158, 469),
    FaceLandmarksConnections.Connection(469, 145),
    FaceLandmarksConnections.Connection(158, 157),
    FaceLandmarksConnections.Connection(157, 469),
    FaceLandmarksConnections.Connection(469, 153),
    FaceLandmarksConnections.Connection(153, 145),
    FaceLandmarksConnections.Connection(157, 153),
    FaceLandmarksConnections.Connection(154, 157),
    FaceLandmarksConnections.Connection(153, 154),
    FaceLandmarksConnections.Connection(157, 173),
    FaceLandmarksConnections.Connection(154, 173),
    FaceLandmarksConnections.Connection(155, 173),
    FaceLandmarksConnections.Connection(154, 155),
    FaceLandmarksConnections.Connection(133, 173),
    FaceLandmarksConnections.Connection(133, 155),
]

_RIGHT_EYE_TESSELATION: list[FaceLandmarksConnections.Connection] = [
    FaceLandmarksConnections.Connection(362, 398),
    FaceLandmarksConnections.Connection(398, 382),
    FaceLandmarksConnections.Connection(382, 362),
    FaceLandmarksConnections.Connection(398, 381),
    FaceLandmarksConnections.Connection(381, 382),
    FaceLandmarksConnections.Connection(398, 384),
    FaceLandmarksConnections.Connection(384, 381),
    FaceLandmarksConnections.Connection(384, 476),
    FaceLandmarksConnections.Connection(476, 381),
    FaceLandmarksConnections.Connection(384, 385),
    FaceLandmarksConnections.Connection(385, 476),
    FaceLandmarksConnections.Connection(476, 380),
    FaceLandmarksConnections.Connection(380, 381),
    FaceLandmarksConnections.Connection(385, 473),
    FaceLandmarksConnections.Connection(473, 476),
    FaceLandmarksConnections.Connection(380, 473),
    FaceLandmarksConnections.Connection(385, 386),
    FaceLandmarksConnections.Connection(386, 473),
    FaceLandmarksConnections.Connection(374, 473),
    FaceLandmarksConnections.Connection(374, 380),
    FaceLandmarksConnections.Connection(386, 387),
    FaceLandmarksConnections.Connection(387, 473),
    FaceLandmarksConnections.Connection(374, 474),
    FaceLandmarksConnections.Connection(473, 474),
    FaceLandmarksConnections.Connection(387, 474),
    FaceLandmarksConnections.Connection(373, 374),
    FaceLandmarksConnections.Connection(373, 474),
    FaceLandmarksConnections.Connection(387, 388),
    FaceLandmarksConnections.Connection(388, 474),
    FaceLandmarksConnections.Connection(390, 474),
    FaceLandmarksConnections.Connection(390, 373),
    FaceLandmarksConnections.Connection(390, 388),
    FaceLandmarksConnections.Connection(388, 466),
    FaceLandmarksConnections.Connection(390, 466),
    FaceLandmarksConnections.Connection(249, 466),
    FaceLandmarksConnections.Connection(249, 390),
    FaceLandmarksConnections.Connection(249, 263),
    FaceLandmarksConnections.Connection(263, 466),
]


_GENERAL_FACE_MESH_LANDMARK_INDEX_TRIANGLES: npt.NDArray[np.int32]


def initialize_triangles() -> None:
    global _GENERAL_FACE_MESH_LANDMARK_INDEX_TRIANGLES

    adjacency: dict[int, set[int]] = {}
    for tesselation in [vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION, _LIPS_INNER_TESSELATION, _LEFT_EYE_TESSELATION, _RIGHT_EYE_TESSELATION]:
        for connection in tesselation:
            adjacency.setdefault(connection.start, set()).add(connection.end)
            adjacency.setdefault(connection.end, set()).add(connection.start)

    triangles: set[tuple[int, int, int]] = set()
    for a, neighbours in adjacency.items():
        for b in neighbours:
            if b <= a:
                continue
            for c in adjacency[a].intersection(adjacency[b]):
                if c <= b:
                    continue
                triangles.add((a, b, c))

    # noinspection PyTypeChecker
    _GENERAL_FACE_MESH_LANDMARK_INDEX_TRIANGLES = np.asarray(sorted(triangles), dtype=np.int32)


@dataclass(frozen=True)
class FaceLandmarker:
    detector: vision.FaceLandmarker
    lock: threading.Lock


def initialize(model_path: str, conf: float) -> FaceLandmarker:
    return FaceLandmarker(
        vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )),
        threading.Lock()
    )


# noinspection PyTypeHints
def detect_get_face_landmarks(face_landmarker: FaceLandmarker, img: mp.Image) -> npt.NDArray[NormalizedLandmark] | None:
    landmarker_result: vision.FaceLandmarkerResult | None = None
    with face_landmarker.lock:
        landmarker_result = face_landmarker.detector.detect(img)
    if landmarker_result is None or len(landmarker_result.face_landmarks) != 1:
        return None
    return np.asarray(landmarker_result.face_landmarks[0])


# noinspection PyTypeHints
def get_landmarks_float_pixel_coords(img_shape, landmarks: npt.NDArray[NormalizedLandmark]) -> npt.NDArray[np.float32]:
    h, w = img_shape[:2]
    h_max = h - 1
    w_max = w - 1

    # noinspection PyTypeChecker
    points: npt.NDArray[np.float32] = np.empty((len(landmarks), 2), dtype=np.float32)

    for i, lm in enumerate(landmarks):
        if lm.x is not None and lm.y is not None:
            px = lm.x * w
            py = lm.y * h

            if px < 0.0:
                px = 0.0
            elif px > w_max:
                px = w_max

            if py < 0.0:
                py = 0.0
            elif py > h_max:
                py = h_max

            points[i][0] = px
            points[i][1] = py
        else:
            points[i][0] = -1.0

    return points


def get_triangle_points(landmarks_float_pixel_coords: npt.NDArray[np.float32], landmark_index_triangle: npt.NDArray[np.int32]) -> npt.NDArray[np.float32] | None:
    triangle_pixel_points = landmarks_float_pixel_coords[landmark_index_triangle]
    if np.any(triangle_pixel_points < 0.0):
        return None
    v1 = triangle_pixel_points[1] - triangle_pixel_points[0]
    v2 = triangle_pixel_points[2] - triangle_pixel_points[0]
    area = float(np.abs(v1[0] * v2[1] - v1[1] * v2[0]))
    if area >= 1e-3:
        return triangle_pixel_points
    return None


def get_triangles_float_pixel_coords(landmarks_float_pixel_coords: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32] | None]:
    out: list[npt.NDArray[np.float32] | None] = [None] * len(_GENERAL_FACE_MESH_LANDMARK_INDEX_TRIANGLES)
    for i, landmark_index_triangle in enumerate(_GENERAL_FACE_MESH_LANDMARK_INDEX_TRIANGLES):
        out[i] = get_triangle_points(landmarks_float_pixel_coords, landmark_index_triangle)
    return out


def get_mesh_mask(img_shape, triangles_float_pixel_coords: list[npt.NDArray[np.float32] | None]) -> npt.NDArray[np.uint8]:
    # noinspection PyTypeChecker
    mask: npt.NDArray[np.uint8] = np.zeros(img_shape[:2], dtype=np.uint8)
    for triangle_float_pixel_coords in triangles_float_pixel_coords:
        if triangle_float_pixel_coords is not None:
            cv2.fillConvexPoly(mask, triangle_float_pixel_coords.astype(np.int32), 255)
    return mask


# noinspection PyTypeHints
def get_annotated_image(img: npt.NDArray[np.uint8], landmarks: npt.NDArray[NormalizedLandmark]) -> npt.NDArray[np.uint8]:
    annotated_image = img[:, :, :3].copy()
    landmarks_list = landmarks.tolist()

    vision.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_list,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    vision.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_list,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_contours_style()
    )
    vision.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_list,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_iris_connections_style()
    )
    vision.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks_list,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_iris_connections_style()
    )

    return annotated_image


def _warp_triangle_add(
    src_img: npt.NDArray[np.uint8], src_tr_pts: npt.NDArray[np.float32], dst_tr_pts: npt.NDArray[np.float32],
    out: npt.NDArray[np.uint8]
) -> None:
    src_tr_pts_int = src_tr_pts.astype(np.int32)
    (sx, sy, sw, sh) = cv2.boundingRect(src_tr_pts_int)
    if sw == 0 or sh == 0:
        return

    dst_tr_pts_int = dst_tr_pts.astype(np.int32)
    (dx, dy, dw, dh) = cv2.boundingRect(dst_tr_pts_int)
    if dw == 0 or dh == 0:
        return

    src_local = src_tr_pts - np.asarray([sx, sy], dtype=np.float32)
    dst_local = dst_tr_pts - np.asarray([dx, dy], dtype=np.float32)
    matrix = cv2.getAffineTransform(src_local, dst_local)
    src_cropped_triangle = src_img[sy:sy+sh, sx:sx+sw]

    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    dst_cropped_triangle_mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(dst_cropped_triangle_mask, dst_local.astype(np.int32), 255)
    dst_cropped_triangle_mask_bool = dst_cropped_triangle_mask.view(np.bool_)
    warped_triangle *= dst_cropped_triangle_mask_bool[..., None]

    (out[dy:dy+dh, dx:dx+dw])[dst_cropped_triangle_mask_bool] = warped_triangle[dst_cropped_triangle_mask_bool]


def warp_face_mesh(
    src_img_rgb: npt.NDArray[np.uint8], src_face_mask: npt.NDArray[np.uint8], dst_shape,
    src_triangles_float_pixel_coords: list[npt.NDArray[np.float32] | None], dst_triangles_float_pixel_coords: list[npt.NDArray[np.float32] | None]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    # noinspection PyTypeChecker
    src_img_rgba: npt.NDArray[np.uint8] = np.empty((src_img_rgb.shape[0], src_img_rgb.shape[1], 4), dtype=np.uint8)
    src_img_rgba[..., :3] = src_img_rgb
    src_img_rgba[..., 3] = src_face_mask

    # noinspection PyTypeChecker
    warped_face: npt.NDArray[np.uint8] = np.zeros((dst_shape[0], dst_shape[1], 4), dtype=np.uint8)

    assert len(src_triangles_float_pixel_coords) == len(dst_triangles_float_pixel_coords)
    for src_triangle_float_pixel_coords, dst_triangle_float_pixel_coords in zip(src_triangles_float_pixel_coords, dst_triangles_float_pixel_coords):
        if src_triangle_float_pixel_coords is None or dst_triangle_float_pixel_coords is None:
            continue
        _warp_triangle_add(src_img_rgba, src_triangle_float_pixel_coords, dst_triangle_float_pixel_coords, warped_face)

    return warped_face[..., :3], warped_face[..., 3]
