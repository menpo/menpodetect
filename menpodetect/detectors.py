from __future__ import division
import numpy as np
from menpo.transform import UniformScale

_bounding_box_adj = np.array([[0, 3], [2, 0], [1, 2], [1, 3]])


def _greyscale(image):
    if image.n_channels > 1:
        if image.n_channels == 3:
            # Use luminosity for RGB images
            image = image.as_greyscale(mode='luminosity')
        else:
            # Fall back to the average of the channels for other kinds
            # of images
            image = image.as_greyscale(mode='average')
    return image


def menpo_image_to_uint8(image):
    return np.array(image.as_PILImage())


def detect(detector_callable, image, greyscale=True,
           image_diagonal=None, group_prefix='object'):
    d_image = image

    if greyscale:
        d_image = _greyscale(d_image)

    if image_diagonal is not None:
        scale_factor = image_diagonal / image.diagonal
        d_image = d_image.rescale(scale_factor)

    pcs = detector_callable(menpo_image_to_uint8(d_image))

    if image_diagonal is not None:
        pcs = [UniformScale(1 / scale_factor, n_dims=2).apply(pc) for pc in pcs]

    padding_magnitude = len(str(len(pcs)))
    for i, pc in enumerate(pcs):
        key = '{prefix}_{num:0{mag}d}'.format(mag=padding_magnitude,
                                              prefix=group_prefix, num=i)
        image.landmarks[key] = pc
    return pcs
