import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import numpy.typing as npt
import cv2
from dataclasses import dataclass
import argparse
from pathlib import Path
from app.detectors import simple_face_detector, complex_landmark_detector, utils


DST_IMAGES_FOLDER: Path = (Path(__file__).parent / 'dst_images').resolve()


def dst_images() -> dict[str, Path]:
    return {
        file_path.stem: file_path
        for file_path in sorted(DST_IMAGES_FOLDER.iterdir())
        if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
    }


@dataclass
class Globals:
    DEBUG_MODE: bool
    WEBP_QUALITY: int = 80
    DST_NAME: str = 'dicaprio'

    MIN_FACE_DETECTION_CONFIDENCE: float = 0.6
    SIMPLE_FACE_CROP_MARGIN_PERCENT: float = 0.5

    DST_SOFT_FACE_MASK_PREBLEND_SIGMA_X: float = 20.0
    DST_SOFT_FACE_MASK_SIGMA_X: float = 17.0

    SRC_FACE_MASK_ERODE: int = 15
    SRC_INPAINTED_BLUR_KSIZE: int = 15


def parse_globals() -> Globals:
    parser = argparse.ArgumentParser()

    parser.add_argument('-debug_mode', action='store_true', default=False, help='Enable temporary debug image outputs. (default: false)')
    parser.add_argument('-webp_quality', type=int, choices=range(1, 101), metavar='[1-100]', default=Globals.WEBP_QUALITY, help=f'Webp quality level. (default: {Globals.WEBP_QUALITY})')
    parser.add_argument('-dst_name', choices=sorted(dst_images()), default=Globals.DST_NAME, help=f'Dst image to warp faces onto. (default: {Globals.DST_NAME})')
    args = parser.parse_args()

    out = Globals(args.debug_mode, args.webp_quality, args.dst_name)
    print(str(out), end='\n\n', flush=True)

    return out


@dataclass(frozen=True)
class OperationResult:
    code: int
    message: str


def read_image(img: npt.NDArray[np.uint8], preserve_alpha_channel: bool) -> mp.Image | OperationResult:
    if img is None or len(img) == 0:
        return OperationResult(415, 'Unsupported Media Type: Unable to read image.')

    img_mp: mp.Image | None = None
    try:
        if img.ndim == 2:
            img_mp = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        elif img.ndim == 3:
            if img.shape[2] == 3:
                img_mp = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif img.shape[2] == 4:
                if preserve_alpha_channel:
                    img_mp = mp.Image(mp.ImageFormat.SRGBA, cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
                else:
                    img_mp = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
    except Exception as e:
        return OperationResult(415, f'Unsupported Media Type: {str(e)}.')

    if img_mp is None:
        return OperationResult(415, 'Unsupported Media Type: Unable to read image.')
    if img_mp.is_empty():
        return OperationResult(415, 'Unsupported Media Type: Unable to read image.')

    return img_mp


def read_image_from_file(file_path: Path, preserve_alpha_channel: bool) -> mp.Image | OperationResult:
    # noinspection PyTypeChecker
    img: npt.NDArray[np.uint8] = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return read_image(img, preserve_alpha_channel)


def read_image_from_bytes(buf: bytes, preserve_alpha_channel: bool) -> mp.Image | OperationResult:
    # noinspection PyTypeChecker
    img: npt.NDArray[np.uint8] = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    return read_image(img, preserve_alpha_channel)


@dataclass(repr=False, eq=False, frozen=True)
class DstData:
    img: npt.NDArray[np.uint8]
    img_flt: npt.NDArray[np.float32]

    triangles_float_pixel_coords: list[npt.NDArray[np.float32] | None]
    center_face: tuple[int, int]
    face_mesh_mask: npt.NDArray[np.uint8]
    face_mesh_mask_bool: npt.NDArray[np.bool_]

    face_mesh_soft_mask_preblend: npt.NDArray[np.float32]
    face_mesh_soft_mask_preblend_inv: npt.NDArray[np.float32]
    face_mesh_soft_mask_blend: npt.NDArray[np.float32]
    face_mesh_soft_mask_blend_inv: npt.NDArray[np.float32]


def load_dst(cached_face_landmarker: vision.FaceLandmarker, gl: Globals, file_path: Path) -> DstData | OperationResult:
    res = read_image_from_file(file_path, True)
    if isinstance(res, OperationResult):
        return res

    face_landmarks = complex_landmark_detector.detect_get_face_landmarks(cached_face_landmarker, res)
    if face_landmarks is None:
        return OperationResult(415, 'Unsupported Media Type: No face landmarks or multiple face landmarks detected (FaceLandmarker).')

    img_np = res.numpy_view().copy()
    img_flt: npt.NDArray[np.float32] = img_np.astype(np.float32)
    landmarks_float_pixel_coords = complex_landmark_detector.get_landmarks_float_pixel_coords(img_np.shape, face_landmarks)
    triangles_float_pixel_coords = complex_landmark_detector.get_triangles_float_pixel_coords(landmarks_float_pixel_coords)
    convex_hull = cv2.convexHull(landmarks_float_pixel_coords)
    (x, y, w, h) = cv2.boundingRect(convex_hull)
    center_face = (int(x + w / 2), int(y + h / 2))
    face_mesh_mask = complex_landmark_detector.get_mesh_mask(img_np.shape, triangles_float_pixel_coords)
    face_mesh_mask_bool = face_mesh_mask.view(np.bool_)
    face_mesh_soft_mask_preblend = utils.get_soft_mask(face_mesh_mask, gl.DST_SOFT_FACE_MASK_PREBLEND_SIGMA_X)
    face_mesh_soft_mask_preblend_inv = 1.0 - face_mesh_soft_mask_preblend
    face_mesh_soft_mask_blend = utils.get_soft_mask(face_mesh_mask, gl.DST_SOFT_FACE_MASK_SIGMA_X)
    face_mesh_soft_mask_blend_inv = 1.0 - face_mesh_soft_mask_blend

    return DstData(
        img_np, img_flt,
        triangles_float_pixel_coords, center_face, face_mesh_mask, face_mesh_mask_bool,
        face_mesh_soft_mask_preblend, face_mesh_soft_mask_preblend_inv, face_mesh_soft_mask_blend, face_mesh_soft_mask_blend_inv
    )


def save_image(path_clean_filename: Path, img: npt.NDArray[np.uint8], webp_quality: int) -> npt.NDArray[np.uint8] | OperationResult:
    if img.ndim == 2:
        pass
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        return OperationResult(415, 'Unsupported Media Type: Unable to save image due to unsupported format.')

    success, encoded = cv2.imencode('.webp', img, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
    if not success:
        return OperationResult(500, 'Internal Server Error: Failed to encode WebP image.')

    with open(f'{path_clean_filename}.webp', "wb") as f:
        if f.write(encoded.tobytes()) != len(encoded):
            return OperationResult(500, 'Internal Server Error: Failed to save WebP image.')

    return encoded


@dataclass(repr=False, eq=False, frozen=True)
class ImageResult:
    name: str
    img: npt.NDArray[np.uint8]


def process_image(
    cached_face_detector: vision.FaceDetector, cached_face_landmarker: vision.FaceLandmarker,
    gl: Globals, src: mp.Image, dst: DstData
) -> list[ImageResult] | OperationResult:
    out: list[ImageResult] = []

    # Detecting face with simple FaceDetector on original image
    face_crop_view = simple_face_detector.detect_get_crop_view(cached_face_detector, src, gl.SIMPLE_FACE_CROP_MARGIN_PERCENT)
    if face_crop_view is None:
        return OperationResult(415, 'Unsupported Media Type: No faces or multiple faces detected (FaceDetector).')
    face_crop_resized = utils.resize(face_crop_view)
    if gl.DEBUG_MODE:
        out.append(ImageResult('0 face crop resized', face_crop_resized))

    # Detecting landmarks with complex FaceLandmarker on face crop
    face_landmarks = complex_landmark_detector.detect_get_face_landmarks(cached_face_landmarker, mp.Image(src.image_format, face_crop_resized))
    if face_landmarks is None:
        return OperationResult(415, 'Unsupported Media Type: No face landmarks or multiple face landmarks detected (FaceLandmarker).')
    if gl.DEBUG_MODE:
        out.append(ImageResult('1 landmarks', complex_landmark_detector.get_annotated_image(face_crop_resized, face_landmarks)))

    # Generating complete face mesh mask
    landmarks_float_pixel_coords = complex_landmark_detector.get_landmarks_float_pixel_coords(face_crop_resized.shape, face_landmarks)
    triangles_float_pixel_coords = complex_landmark_detector.get_triangles_float_pixel_coords(landmarks_float_pixel_coords)
    face_mesh_mask = complex_landmark_detector.get_mesh_mask(face_crop_resized.shape, triangles_float_pixel_coords)
    if gl.DEBUG_MODE:
        face_mesh_extracted = face_crop_resized.copy()
        face_mesh_mask_bool = face_mesh_mask.view(np.bool_)
        face_mesh_extracted *= face_mesh_mask_bool[..., None]
        out.append(ImageResult('2 mesh extracted', face_mesh_extracted.copy()))

    # Eroding face mesh mask
    utils.erode_mask_inplace(face_mesh_mask, gl.SRC_FACE_MASK_ERODE)
    if gl.DEBUG_MODE:
        face_mesh_extracted = face_crop_resized.copy()
        face_mesh_mask_bool = face_mesh_mask.view(np.bool_)
        face_mesh_extracted *= face_mesh_mask_bool[..., None]
        out.append(ImageResult('3 mesh mask eroded', face_mesh_extracted.copy()))

    # Warping face mesh to dst face mesh
    face_warped, face_warped_mask = complex_landmark_detector.warp_face_mesh(face_crop_resized, face_mesh_mask, dst.img.shape, triangles_float_pixel_coords, dst.triangles_float_pixel_coords)
    if gl.DEBUG_MODE:
        out.append(ImageResult('4 warped', face_warped.copy()))
        out.append(ImageResult('5 warped mask', face_warped_mask.copy()))

    # Inpainting to dst oval contour
    face_warped_mask_bool = face_warped_mask.view(np.bool_)
    fill_mask_bool = utils.fill_from_nearest_valid_pixels_inplace(face_warped, face_warped_mask_bool, dst.face_mesh_mask_bool)
    if gl.DEBUG_MODE:
        out.append(ImageResult('6 in-painted', face_warped.copy()))

    # Blur in-painted data
    utils.blur_mask_inplace(face_warped, fill_mask_bool, gl.SRC_INPAINTED_BLUR_KSIZE)
    if gl.DEBUG_MODE:
        out.append(ImageResult('7 blurred', face_warped.copy()))

    # Match color to dst image
    face_warped_colored_lab = utils.match_color_lab(face_warped, dst.img, dst.face_mesh_mask_bool)
    face_warped_colored = utils.match_color(face_warped_colored_lab, dst.img, dst.face_mesh_mask_bool)
    if gl.DEBUG_MODE:
        out.append(ImageResult('8 LAB and match color', face_warped_colored))

    # Seamlessly clone warped face to dst image
    preblended_face = utils.blend_rgb(face_warped_colored, dst.img_flt, dst.face_mesh_soft_mask_preblend, dst.face_mesh_soft_mask_preblend_inv)
    result = utils.swap_new_face(dst.img, dst.center_face, preblended_face, dst.face_mesh_mask)
    if gl.DEBUG_MODE:
        out.append(ImageResult('9 seamless clone', result.copy()))

    # Blend inserted face edges with dst image
    result[..., :3] = utils.blend_rgb(result, dst.img_flt, dst.face_mesh_soft_mask_blend, dst.face_mesh_soft_mask_blend_inv)

    out.append(ImageResult('10 edge blend (result)', result))
    return out
