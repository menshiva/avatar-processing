from mediapipe import Image
import shutil
from pathlib import Path
from app.detectors import simple_face_detector, complex_landmark_detector
from app import processor


APP_FOLDER: Path = Path(__file__).parent.resolve()
SRC_FOLDER: Path = APP_FOLDER / 'local_images' / 'in'
TMP_FOLDER: Path = APP_FOLDER / 'local_images' / 'tmp'
RES_FOLDER: Path = APP_FOLDER / 'local_images' / 'out'


def load_input_images() -> list[tuple[Path, Image]]:
    print('Loading input images...', flush=True)

    if not SRC_FOLDER.exists():
        print(f'ERROR: {SRC_FOLDER} folder does not exist.', flush=True)
        return []

    images: list[tuple[Path, Image]] = []
    for file_path in SRC_FOLDER.iterdir():
        if not file_path.is_file():
            continue
        print(f'\tLoading {file_path.name}... ', end='', flush=True)

        mp_image = processor.read_image_from_file(file_path, False)
        if isinstance(mp_image, processor.OperationResult):
            print(f'ERROR ({mp_image.message}).', flush=True)
            continue

        images.append((file_path, mp_image))
        print('OK.', flush=True)

    if len(images) > 0:
        print(f'OK. Parsed {len(images)} files.', flush=True)
    else:
        print(f'ERROR: No valid images found in the {SRC_FOLDER} folder.', flush=True)

    return images


def main() -> None:
    gl = processor.parse_globals()

    print('Precomputing face mesh triangles... ', end='', flush=True)
    complex_landmark_detector.initialize_triangles()
    print('OK.')
    print('', flush=True)

    print('Initializing simple FaceDetector for face detection...', flush=True)
    face_detector = simple_face_detector.initialize(str(APP_FOLDER / 'models' / 'blaze_face_full_range_sparse.tflite'), gl.MIN_FACE_DETECTION_CONFIDENCE)
    print('OK.', flush=True)
    print('Initializing FaceLandmarker for facial landmark detection...', flush=True)
    face_landmarker = complex_landmark_detector.initialize(str(APP_FOLDER / 'models' / 'face_landmarker_v2_with_blendshapes.task'), gl.MIN_FACE_DETECTION_CONFIDENCE)
    print('OK.')
    print('', flush=True)

    print(f'Preparing dst image ({gl.DST_NAME})... ', end='', flush=True)
    dst = processor.load_dst(face_landmarker, gl, processor.dst_images()[gl.DST_NAME])
    if isinstance(dst, processor.OperationResult):
        print(f'ERROR ({dst.message}).', flush=True)
        return
    print('OK.')
    print('', flush=True)

    images = load_input_images()
    if len(images) == 0:
        return
    print('', flush=True)

    shutil.rmtree(TMP_FOLDER, ignore_errors=True)
    shutil.rmtree(RES_FOLDER, ignore_errors=True)
    if gl.DEBUG_MODE:
        TMP_FOLDER.mkdir(parents=True, exist_ok=True)
    RES_FOLDER.mkdir(parents=True, exist_ok=True)

    for file_path, image in images:
        print(f'Processing {file_path.stem}{file_path.suffix}... ', end='', flush=True)

        res = processor.process_image(face_detector, face_landmarker, gl, image, dst)
        if isinstance(res, processor.OperationResult):
            print(f'ERROR ({res.message}).', flush=True)
            continue

        if gl.DEBUG_MODE:
            for img_res in res:
                processor.save_image(TMP_FOLDER / f'{file_path.stem} - {img_res.name}', img_res.img, gl.WEBP_QUALITY)
        processor.save_image(RES_FOLDER / file_path.stem, res[-1].img, gl.WEBP_QUALITY)

        print('OK.', flush=True)


if __name__ == '__main__':
    main()
