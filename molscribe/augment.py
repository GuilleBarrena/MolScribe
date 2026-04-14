import albumentations as A
from albumentations.augmentations.geometric.functional import pad_with_params
import cv2
import math
import random
import numpy as np


def _safe_rotate_enlarged_img_size(angle: float, rows: int, cols: int):
    """Compute bounding box dimensions of a rotated rectangle. Ported from albumentations 1.1.0."""
    sin = abs(math.sin(math.radians(angle)))
    cos = abs(math.cos(math.radians(angle)))
    new_rows = int(math.ceil(rows * cos + cols * sin))
    new_cols = int(math.ceil(rows * sin + cols * cos))
    return new_rows, new_cols


def _keypoint_rotate(keypoint, angle: float, rows: int, cols: int):
    """Rotate a keypoint around the image center. Ported from albumentations 1.1.0."""
    center = ((cols - 1) * 0.5, (rows - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()[:2]
    return x, y, a + math.radians(angle), s


def safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = _safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # rotate image with the new bounds
    rotated_img = cv2.warpAffine(
        img,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    return rotated_img


def keypoint_safe_rotate(keypoint, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = _safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = (new_cols - old_cols) / 2
    row_diff = (new_rows - old_rows) / 2

    # Shift keypoint
    shifted_keypoint = (int(keypoint[0] + col_diff), int(keypoint[1] + row_diff), keypoint[2], keypoint[3])

    # Rotate keypoint
    rotated_keypoint = _keypoint_rotate(shifted_keypoint, angle, rows=new_rows, cols=new_cols)

    return rotated_keypoint


class SafeRotate(A.SafeRotate):

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        p=0.5,
    ):
        # albumentations 1.4.x uses fill/fill_mask instead of value/mask_value
        super(SafeRotate, self).__init__(
            limit=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            fill=value if value is not None else 0,
            fill_mask=mask_value if mask_value is not None else 0,
            p=p)
        self._fill_value = value

    def apply(self, img, rotate=0, **params):
        # 'rotate' is the key emitted by SafeRotate.get_params_dependent_on_data in albumentations 1.4.x
        return safe_rotate(
            img=img, value=self._fill_value, angle=rotate, interpolation=self.interpolation,
            border_mode=self.border_mode)

    def apply_to_keypoints(self, keypoints: np.ndarray, rotate=0, **params):
        # keypoints: np.ndarray shape (N, 5) — columns [x, y, angle, scale, extra]
        rows, cols = params["shape"][:2]
        result = keypoints.copy()
        for i, kp in enumerate(keypoints):
            x, y, a, s = keypoint_safe_rotate(tuple(kp[:4]), angle=rotate, rows=rows, cols=cols)
            result[i, 0] = x
            result[i, 1] = y
            result[i, 2] = a
            result[i, 3] = s
        return result


class CropWhite(A.DualTransform):

    def __init__(self, value=(255, 255, 255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        img = pad_with_params(
            img, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, self.value)
        return img

    def apply_to_keypoints(self, keypoints: np.ndarray, crop_top=0, crop_bottom=0,
                            crop_left=0, crop_right=0, **params):
        result = keypoints.copy()
        result[:, 0] = result[:, 0] - crop_left + self.pad
        result[:, 1] = result[:, 1] - crop_top + self.pad
        return result

    def get_transform_init_args_names(self):
        return ('value', 'pad')


class PadWhite(A.DualTransform):

    def __init__(self, pad_ratio=0.2, p=0.5, value=(255, 255, 255)):
        super(PadWhite, self).__init__(p=p)
        self.pad_ratio = pad_ratio
        self.value = value

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        side = random.randrange(4)
        if side == 0:
            params['pad_top'] = int(height * self.pad_ratio * random.random())
        elif side == 1:
            params['pad_bottom'] = int(height * self.pad_ratio * random.random())
        elif side == 2:
            params['pad_left'] = int(width * self.pad_ratio * random.random())
        elif side == 3:
            params['pad_right'] = int(width * self.pad_ratio * random.random())
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        img = pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, self.value)
        return img

    def apply_to_keypoints(self, keypoints: np.ndarray, pad_top=0, pad_bottom=0,
                            pad_left=0, pad_right=0, **params):
        result = keypoints.copy()
        result[:, 0] = result[:, 0] + pad_left
        result[:, 1] = result[:, 1] + pad_top
        return result

    def get_transform_init_args_names(self):
        return ('value', 'pad_ratio')


class SaltAndPepperNoise(A.DualTransform):

    def __init__(self, num_dots, value=(0, 0, 0), p=0.5):
        super().__init__(p)
        self.num_dots = num_dots
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        num_dots = random.randrange(self.num_dots + 1)
        for i in range(num_dots):
            x = random.randrange(height)
            y = random.randrange(width)
            img[x, y] = self.value
        return img

    def apply_to_keypoints(self, keypoints: np.ndarray, **params):
        return keypoints

    def get_transform_init_args_names(self):
        return ('value', 'num_dots')


class ResizePad(A.DualTransform):

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, value=(255, 255, 255)):
        super(ResizePad, self).__init__(p=1)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.value = value

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _ = img.shape
        target_h = min(h, self.height)
        target_w = min(w, self.width)
        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        h, w, _ = img.shape
        pad_top = (self.height - h) // 2
        pad_bottom = (self.height - h) - pad_top
        pad_left = (self.width - w) // 2
        pad_right = (self.width - w) - pad_left
        img = pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, self.value)
        return img


def normalized_grid_distortion(
        img,
        num_steps=10,
        xsteps=(),
        ysteps=(),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
):
    height, width = img.shape[:2]

    # compensate for smaller last steps in source image.
    x_step = width // num_steps
    last_x_step = min(width, ((num_steps + 1) * x_step)) - (num_steps * x_step)
    xsteps[-1] *= last_x_step / x_step

    y_step = height // num_steps
    last_y_step = min(height, ((num_steps + 1) * y_step)) - (num_steps * y_step)
    ysteps[-1] *= last_y_step / y_step

    # now normalize such that distortion never leaves image bounds.
    tx = width / math.floor(width / num_steps)
    ty = height / math.floor(height / num_steps)
    xsteps = np.array(xsteps) * (tx / np.sum(xsteps))
    ysteps = np.array(ysteps) * (ty / np.sum(ysteps))

    # build remap grids (inlined from albumentations 1.1.0 grid_distortion)
    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]
        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=value)


class NormalizedGridDistortion(A.GridDistortion):
    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        return normalized_grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation,
                                          self.border_mode, self.value)

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return normalized_grid_distortion(img, self.num_steps, stepsx, stepsy,
                                          cv2.INTER_NEAREST, self.border_mode, self.mask_value)
