from __future__ import division
import numpy as np
from menpo.transform import UniformScale


def _greyscale(image):
    r"""
    Convert image to greyscale if needed. If the image has more than 3 channels,
    then the average greyscale is taken. A copy of the image as greyscale is
    returned (single channel).

    Parameters
    ----------
    image : `menpo.image.Image`
        The image to convert.

    Returns
    -------
    image : `menpo.image.Image`
        A greyscale version of the image.
    """
    if image.n_channels != 1:
        if image.n_channels == 3:
            # Use luminosity for RGB images
            image = image.as_greyscale(mode='luminosity')
        else:
            # Fall back to the average of the channels for other kinds
            # of images
            image = image.as_greyscale(mode='average')
    return image


def menpo_image_to_uint8(image, channels_at_back=True):
    r"""
    Return the given image as a uint8 array. This is a copy of the image.

    Parameters
    ----------
    image : `menpo.image.Image`
        The image to convert. If already uint8, only the channels will be
        rolled to the last axis.
    channels_at_back : `bool`, optional
        If ``True``, the image channels are placed onto the last axis (the back)
        as is common in many imaging packages. This is contrary to the Menpo
        default where channels are the first axis (at the front).

    Returns
    -------
    uint8_image : `ndarray`
        `uint8` Numpy array, channels as the back (last) axis if
        ``channels_at_back == True``.
    """
    if channels_at_back:
        uint8_im = image.pixels_with_channels_at_back(out_dtype=np.uint8)
        # Handle the dead axis on greyscale images
        if uint8_im.ndim == 3 and uint8_im.shape[-1] == 1:
            uint8_im = uint8_im[..., 0]
    else:
        from menpo.image.base import denormalize_pixels_range
        uint8_im = denormalize_pixels_range(image.pixels, np.uint8)
        # Handle the dead axis on greyscale images
        if uint8_im.ndim == 3 and uint8_im.shape[0] == 1:
            uint8_im = uint8_im[0]
    return uint8_im


def detect(detector_callable, image, greyscale=True,
           image_diagonal=None, group_prefix='object', channels_at_back=True):
    r"""
    Apply the general detection framework.

    This involves converting the image to greyscale if necessary, rescaling
    the image to a given diagonal, performing the detection, and attaching
    the scaled landmarks back onto the original image.

    uint8 images cannot be converted to greyscale by this framework, so must
    already be greyscale or ``greyscale=False``.

    Parameters
    ----------
    detector_callable : `callable` or `function`
        A callable object that will perform detection given a single parameter,
        a `uint8` numpy array with either no channels, or channels as the
        *last* axis.
    image : `menpo.image.Image`
        A Menpo image to detect. The bounding boxes of the detected objects
        will be attached to this image.
    greyscale : `bool`, optional
        Convert the image to greyscale or not.
    image_diagonal : `int`, optional
        The total size of the diagonal of the image that should be used for
        detection. This is useful for scaling images up and down for detection.
    group_prefix : `str`, optional
        The prefix string to be appended to each each landmark group that is
        stored on the image. Each detection will be stored as group_prefix_#
        where # is a count starting from 0.
    channels_at_back : `bool`, optional
        If ``True``, the image channels are placed onto the last axis (the back)
        as is common in many imaging packages. This is contrary to the Menpo
        default where channels are the first axis (at the front).

    Returns
    -------
    bounding_boxes : `list` of `menpo.shape.PointDirectedGraph`
        A list of bounding boxes representing the detections found.
    """
    d_image = image

    if greyscale:
        d_image = _greyscale(d_image)

    if image_diagonal is not None:
        scale_factor = image_diagonal / image.diagonal()
        d_image = d_image.rescale(scale_factor)

    pcs = detector_callable(menpo_image_to_uint8(
        d_image, channels_at_back=channels_at_back))

    if image_diagonal is not None:
        s = UniformScale(1 / scale_factor, n_dims=2)
        pcs = [s.apply(pc) for pc in pcs]

    padding_magnitude = len(str(len(pcs)))
    for i, pc in enumerate(pcs):
        key = '{prefix}_{num:0{mag}d}'.format(mag=padding_magnitude,
                                              prefix=group_prefix, num=i)
        image.landmarks[key] = pc
    return pcs
