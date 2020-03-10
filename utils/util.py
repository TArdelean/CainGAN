import os

from PIL import Image

from options import get_options
import numpy as np

opt = get_options()


# Borrows from https://github.com/NVIDIA/pix2pixHD
def tensor2im(image_tensor, imtype=np.uint8, normalize=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, final_size=None):
    image_pil = Image.fromarray(image_numpy)
    if final_size is not None:
        image_pil.resize(final_size)
    image_pil.save(image_path)


def create_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)
