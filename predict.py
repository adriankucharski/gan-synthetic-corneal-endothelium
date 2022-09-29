import json
import sys
from glob import glob
from skimage import io
from scipy.ndimage import gaussian_filter
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Model, load_model
from typing import Tuple, Union
from model import dice_loss


class UnetPrediction:
    def __init__(
        self,
        model_path: str,
        patch_size: int = 64,
        stride: int = 4,
        batch_size: int = 64,
        sigma=None,
    ):
        self.model: Model = load_model(
            model_path, custom_objects={"dice_loss": dice_loss}
        )
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.sigma = sigma

    def _build_img_from_patches(
        self, preds: np.ndarray, img_h: int, img_w: int
    ) -> np.ndarray:
        patch_h, patch_w = preds.shape[1:3]

        H = (img_h - patch_h) // self.stride + 1
        W = (img_w - patch_w) // self.stride + 1
        prob = np.zeros((img_h, img_w, 1))
        _sum = np.zeros((img_h, img_w, 1))

        k = 0
        for h in range(H):
            for w in range(W):
                prob[
                    h * self.stride : (h * self.stride) + patch_h,
                    w * self.stride : (w * self.stride) + patch_w,
                    :,
                ] += preds[k]
                _sum[
                    h * self.stride : (h * self.stride) + patch_h,
                    w * self.stride : (w * self.stride) + patch_w,
                    :,
                ] += 1
                k += 1
        final_avg = prob / _sum
        return final_avg

    def _get_patches(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]

        H = (h - self.patch_size) // self.stride + 1
        W = (w - self.patch_size) // self.stride + 1

        patches = np.empty((W * H, self.patch_size, self.patch_size, 1))
        iter_tot = 0
        for h in range(H):
            for w in range(W):
                patches[iter_tot] = img[
                    h * self.stride : (h * self.stride) + self.patch_size,
                    w * self.stride : (w * self.stride) + self.patch_size,
                    :,
                ]
                iter_tot += 1
        return patches

    def _add_outline(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        over_h = (h - self.patch_size) % self.stride
        over_w = (w - self.patch_size) % self.stride

        if over_h != 0:
            tmp = np.zeros((h + (self.stride - over_h), w, 1))
            tmp[0:h, 0:w, :] = img
            img = tmp
        if over_w != 0:
            tmp = np.zeros((img.shape[0], w + (self.stride - over_w), 1))
            tmp[0 : img.shape[0], 0:w] = img
            img = tmp
        return img

    def _predict_from_array(
        self, data: Tuple[np.ndarray], verbose=0
    ) -> Tuple[np.ndarray]:
        predicted = []
        for idx in range(len(data)):
            height, width = data[idx].shape[:2]
            img = self._add_outline(data[idx])
            if self.sigma is not None:
                img = gaussian_filter(img, self.sigma)
            new_height, new_width = img.shape[:2]
            patches = self._get_patches(img)
            predictions = self.model.predict(
                patches, batch_size=self.batch_size, verbose=verbose
            )
            pred_img = self._build_img_from_patches(predictions, new_height, new_width)

            pred_img = pred_img[:height, :width]
            predicted.append(pred_img)
        return predicted

    def _predict_from_path(
        self, path: Union[str, Tuple[str]], verbose=0
    ) -> Tuple[np.ndarray]:
        images = []

        if isinstance(path, (list, tuple)):
            for im_path in path:
                images.append(io.imread(im_path, as_gray=True))
        if os.path.isfile(path):
            images.append(io.imread(path, as_gray=True))
        if os.path.isdir(path):
            for im_path in glob(os.path.join(path, "*")):
                images.append(io.imread(im_path, as_gray=True))
        for i in range(len(images)):
            if len(images[i].shape) != 3:
                images[i] = images[i][..., np.newaxis]
                if images[i].max() > 1.0:
                    images[i] = images[i] / 255.0
        return self._predict_from_array(images, verbose)

    def predict(
        self, data: Union[np.ndarray, Tuple[np.ndarray], str], verbose=0
    ) -> Tuple[np.ndarray]:
        if isinstance(data, str):
            return self._predict_from_path(data, verbose)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 4:
                return self._predict_from_array(data, verbose)
            elif len(data.shape) == 3:
                return self._predict_from_array([data], verbose)
        if isinstance(data, (list, tuple)) and len(data) > 0:
            if isinstance(data[0], str):
                return self._predict_from_path(data, verbose)
            if isinstance(data[0], np.ndarray):
                return self._predict_from_array(data, verbose)


class GANsPrediction(UnetPrediction):
    def __init__(
        self,
        model_path: str,
        patch_size: int = 64,
        stride: int = 4,
        batch_size: int = 64,
        sigma=None,
    ):
        self.model: Model = load_model(model_path)
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

    def _predict_from_array(
        self, data: Tuple[np.ndarray], verbose=0
    ) -> Tuple[np.ndarray]:
        predicted = []
        for idx in range(len(data)):
            height, width = data[idx].shape[:2]
            img = self._add_outline(data[idx])
            new_height, new_width = img.shape[:2]
            patches = self._get_patches(img)
            predictions = self.model.predict(
                patches, batch_size=self.batch_size, verbose=verbose
            )
            pred_img = self._build_img_from_patches(predictions, new_height, new_width)

            pred_img = pred_img[:height, :width]
            predicted.append(pred_img)
        return predicted


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 4:
        print("Provide at least four arguments")
        exit()
    datasets_names = ["Alizarine", "Gavet", "Hard", "Rotterdam", "Rotterdam_1000"]

    dataset_path, fold, model_path, stride = args[0:4]

    assert dataset_path in datasets_names
    unet_pred = UnetPrediction(model_path, stride=int(stride), batch_size=128)

    dataset_path = f"datasets/{dataset_path}/folds.json"
    with open(dataset_path, "r") as f:
        folds_json = json.load(f)

    if "raw" in model_path:
        prefix = "raw"
    else:
        prefix = "synthetic"

    result_save = Path("./result_image", prefix, folds_json["dataset_name"])
    print(str(result_save))
    result_save.mkdir(parents=True, exist_ok=True)

    parent_path = folds_json["dataset_path"]
    for image in folds_json["folds"][int(fold)]["test"]:
        pr = str(Path(parent_path, "images", image))
        ps = str(Path(result_save, image))
        pred = unet_pred.predict(pr)[0]
        io.imsave(ps, pred)
