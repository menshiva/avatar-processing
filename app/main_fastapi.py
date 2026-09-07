from pathlib import Path
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from starlette.responses import FileResponse
from app.detectors import simple_face_detector, complex_landmark_detector
from app import processor


APP_FOLDER: Path = Path(__file__).parent.resolve()
UPLOADS_FOLDER: Path = APP_FOLDER / 'fastapi_images' / 'uploads'
RES_FOLDER: Path = APP_FOLDER / 'fastapi_images' / 'results'


GLOBALS: processor.Globals
FACE_DETECTOR: simple_face_detector.FaceDetector
FACE_LANDMARKER: complex_landmark_detector.FaceLandmarker
DSTS: dict[str, processor.DstData]


def create_app() -> FastAPI:
    global GLOBALS, FACE_DETECTOR, FACE_LANDMARKER, DSTS

    GLOBALS = processor.Globals(False)
    print(str(GLOBALS), end='\n\n', flush=True)

    print('Precomputing face mesh triangles... ', end='', flush=True)
    complex_landmark_detector.initialize_triangles()
    print('OK.')
    print('', flush=True)

    print('Initializing simple FaceDetector for face detection...', flush=True)
    FACE_DETECTOR = simple_face_detector.initialize(str(APP_FOLDER / 'models' / 'blaze_face_full_range_sparse.tflite'), GLOBALS.MIN_FACE_DETECTION_CONFIDENCE)
    print('OK.', flush=True)
    print('Initializing FaceLandmarker for facial landmark detection...', flush=True)
    FACE_LANDMARKER = complex_landmark_detector.initialize(str(APP_FOLDER / 'models' / 'face_landmarker_v2_with_blendshapes.task'), GLOBALS.MIN_FACE_DETECTION_CONFIDENCE)
    print('OK.')
    print('', flush=True)

    # Every dst image is prepared once here, so a request only pays for the incoming photo
    print('Preparing dst images...', flush=True)
    DSTS = {}
    for name, file_path in processor.dst_images().items():
        print(f'\tPreparing {name}... ', end='', flush=True)
        dst = processor.load_dst(FACE_LANDMARKER, GLOBALS, file_path)
        if isinstance(dst, processor.OperationResult):
            print(f'ERROR ({dst.message}).', flush=True)
            raise RuntimeError('Failed')
        DSTS[name] = dst
        print('OK.', flush=True)
    print(f'OK. Prepared {len(DSTS)} dst images.')
    print('', flush=True)

    UPLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
    RES_FOLDER.mkdir(parents=True, exist_ok=True)

    return FastAPI()


app = create_app()


# noinspection PyUnboundLocalVariable
@app.post('/')
def process_endpoint(
    user_id: str = Form(...),
    dst_name: str = Form(default=GLOBALS.DST_NAME),
    webp_quality: int = Form(default=GLOBALS.WEBP_QUALITY, ge=1, le=100),
    img_file: UploadFile = File(...)
):
    user_id = user_id.strip()
    if user_id == '':
        raise HTTPException(400, 'user_id should not be empty.')

    if user_id in ('.', '..') or user_id != Path(user_id).name:
        raise HTTPException(400, 'user_id should be a plain file name.')

    dst = DSTS.get(dst_name)
    if dst is None:
        raise HTTPException(400, f'Unknown dst_name. Should be one of: {", ".join(DSTS)}. Got {dst_name}.')

    ALLOWED_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'image/webp'}
    if img_file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f'Unsupported Media Type. Should be one of: {", ".join(ALLOWED_TYPES)}. Got {img_file.content_type}.')

    img_bytes = img_file.file.read()
    image = processor.read_image_from_bytes(img_bytes, False)
    if isinstance(image, processor.OperationResult):
        raise HTTPException(image.code, image.message)

    res = processor.process_image(FACE_DETECTOR, FACE_LANDMARKER, GLOBALS, image, dst)
    if isinstance(res, processor.OperationResult):
        raise HTTPException(res.code, res.message)

    processor.save_image(UPLOADS_FOLDER / user_id, image.numpy_view(), webp_quality)
    processor.save_image(RES_FOLDER / f'{user_id}_{dst_name}', res[-1].img, webp_quality)

    return {
        'user_id': user_id,
        'path': f'{user_id}_{dst_name}.webp'
    }


@app.get('/{path}')
def get_image(path: str):
    file_path = (RES_FOLDER / path).resolve()

    if RES_FOLDER not in file_path.parents:
        raise HTTPException(400, 'Invalid path')
    if not file_path.is_file():
        raise HTTPException(404, 'File not found')

    return FileResponse(file_path, media_type="image/webp")
