import numpy as np
import numpy.typing as npt
import cv2


def get_soft_mask(mask: npt.NDArray[np.uint8], sigma_x: float) -> npt.NDArray[np.float32]:
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma_x)
    return soft_mask[..., None]


def resize(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w = image.shape[:2]
    ratio = h / w
    return cv2.resize(image, (512, int(float(512) * ratio)))


def erode_mask_inplace(mask: npt.NDArray[np.uint8], edge_erode_width: int) -> None:
    kernel = np.ones((edge_erode_width, edge_erode_width), np.uint8)
    cv2.erode(mask, kernel, mask, iterations=1)


def fill_from_nearest_valid_pixels_inplace(img: npt.NDArray[np.uint8], face_mask_bool: npt.NDArray[np.bool_], dst_face_mask_bool: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    fill_mask_bool = np.bitwise_not(face_mask_bool)
    np.bitwise_and(fill_mask_bool, dst_face_mask_bool, out=fill_mask_bool)

    region = fill_mask_bool | face_mask_bool
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return fill_mask_bool

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    sub_img = img[y0:y1, x0:x1]
    sub_fill = fill_mask_bool[y0:y1, x0:x1]
    sub_trusted = face_mask_bool[y0:y1, x0:x1]

    _fill_from_nearest_valid_pixels_local(sub_img, sub_fill, sub_trusted)
    return fill_mask_bool


def _fill_from_nearest_valid_pixels_local(img: npt.NDArray[np.uint8], fill_mask: npt.NDArray[np.bool_], trusted_mask: npt.NDArray[np.bool_]) -> None:
    if not np.any(fill_mask) or not np.any(trusted_mask):
        return

    h, w = fill_mask.shape

    # coordinate maps
    yy, xx = np.indices((h, w), dtype=np.int32)

    nearest_y = np.full((h, w), -1, np.int32)
    nearest_x = np.full((h, w), -1, np.int32)

    nearest_y[trusted_mask] = yy[trusted_mask]
    nearest_x[trusted_mask] = xx[trusted_mask]

    kernel = np.ones((3, 3), np.uint8)

    remaining = fill_mask.copy()
    while np.any(remaining):
        known = nearest_x >= 0
        dilated_known = cv2.dilate(known.astype(np.uint8), kernel, iterations=1).astype(bool)

        new_pixels = np.bitwise_not(known)
        np.bitwise_and(new_pixels, dilated_known, out=new_pixels)
        np.bitwise_and(new_pixels, remaining, out=new_pixels)

        if not np.any(new_pixels):
            break

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted_known = np.roll(known, (dy, dx), axis=(0, 1))
                valid = new_pixels & shifted_known

                if not np.any(valid):
                    continue

                shifted_y = np.roll(nearest_y, (dy, dx), axis=(0, 1))
                shifted_x = np.roll(nearest_x, (dy, dx), axis=(0, 1))

                nearest_y[valid] = shifted_y[valid]
                nearest_x[valid] = shifted_x[valid]

        remaining[new_pixels] = False

    valid = nearest_x >= 0
    img[valid] = img[nearest_y[valid], nearest_x[valid]]


def blur_mask_inplace(img: npt.NDArray[np.uint8], mask_bool: npt.NDArray[np.bool_], ksize: int) -> None:
    blurred_rgb = cv2.GaussianBlur(img, (ksize, ksize), 0)
    img[mask_bool] = blurred_rgb[mask_bool]


def match_color_lab(src_img: npt.NDArray[np.uint8], dst_img: npt.NDArray[np.uint8], mask_bool: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    if not np.any(mask_bool):
        return src_img

    # RGB -> LAB
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_img[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)

    result_lab = src_lab.copy()

    # L channel = brightness
    # A/B = color tint
    #
    # We usually want:
    # - stronger luminance matching
    # - softer chroma matching

    channel_strength = [1.0, 0.35, 0.35]

    for c in range(3):
        src_vals = src_lab[..., c][mask_bool]
        dst_vals = dst_lab[..., c][mask_bool]

        src_mean = float(src_vals.mean())
        src_std = float(src_vals.std())

        dst_mean = float(dst_vals.mean())
        dst_std = float(dst_vals.std())

        if src_std < 1e-6:
            continue

        matched = ((result_lab[..., c] - src_mean) * (dst_std / src_std) + dst_mean)
        strength = channel_strength[c]
        result_lab[..., c] = (result_lab[..., c] * (1.0 - strength) + matched * strength)

    # Clamp LAB ranges
    result_lab[..., 0] = np.clip(result_lab[..., 0], 0, 255)
    result_lab[..., 1] = np.clip(result_lab[..., 1], 0, 255)
    result_lab[..., 2] = np.clip(result_lab[..., 2], 0, 255)

    # LAB -> RGB
    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def match_color(src_img: npt.NDArray[np.uint8], dst_img: npt.NDArray[np.uint8], mask_bool: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    result = src_img.copy().astype(np.float32)

    for c in range(3):
        src_vals = src_img[..., c][mask_bool]
        dst_vals = dst_img[..., c][mask_bool]

        src_mean, src_std = src_vals.mean(), src_vals.std()
        dst_mean, dst_std = dst_vals.mean(), dst_vals.std()

        if src_std < 1e-6:
            continue

        result[..., c] = ((result[..., c] - src_mean) * (dst_std / src_std) + dst_mean)

    return np.clip(result, 0, 255).astype(np.uint8)


def blend_rgb(src_img: npt.NDArray[np.uint8], dst_img_flt: npt.NDArray[np.float32], mask: npt.NDArray[np.float32], mask_inv: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    return (src_img[..., :3].astype(np.float32) * mask + dst_img_flt[..., :3] * mask_inv).astype(np.uint8)


def swap_new_face(dst_img: npt.NDArray[np.uint8], center_face: tuple[int, int], new_face: npt.NDArray[np.uint8], face_mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return cv2.seamlessClone(new_face, dst_img, face_mask.copy(), center_face, cv2.NORMAL_CLONE)
