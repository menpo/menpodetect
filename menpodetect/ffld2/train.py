import numpy as np

from menpo.base import MenpoMissingDependencyError

try:
    from cyffld2 import train_model
except ImportError:
    raise MenpoMissingDependencyError('cyffld2')

from menpodetect.detect import menpo_image_to_uint8
from .conversion import ensure_channel_axis


def train_ffld2_detector(positive_images, negative_images, n_components=3,
                         pad_x=6, pad_y=6, interval=5, n_relabel=8,
                         n_datamine=10, max_negatives=24000, C=0.002, J=2.0,
                         overlap=0.5):
    r"""
    Train a DPM using the FFLD2 framework. This is a fairly slow process to
    expect to wait for a while. FFLD2 prints out information at each iteration
    but this will not appear in an IPython notebook, so it is best to run
    this kind of training from the command line.

    This method requires an explicit set of negative images to learn the
    classifier with. The non person images from Pascal VOC 2007 are a good
    example of negative images to train with.

    Parameters
    ----------
    positive_images : `list` of `menpo.image.Image`
        The set of images to learn the detector from. Must have landmarks
        attached to **every** image, a bounding box will be extracted for each
        landmark group.
    negative_images : `list` of `menpo.image.Image`
        The set of images to learn the negative samples of the detector with.
        **No** landmarks need to be attached.
    n_components : `int`
        Number of mixture components (without symmetry).
    pad_x : `int`
        Amount of zero padding in HOG cells (x-direction).
    pad_y : `int`
        Amount of zero padding in HOG cells (y-direction).
    interval : `int`
        Number of levels per octave in the HOG pyramid.
    n_relabel : `int`
        Maximum number of training iterations.
    n_datamine : `int`
        Maximum number of data-mining iterations within each training iteration.
    max_negatives : `int`
        Maximum number of negative images to consider, can be useful for
        reducing training time.
    C : `double`
        SVM regularization constant.
    J : `double`
        SVM positive regularization constant boost.
    overlap : `double`
        Minimum overlap in in latent positive search and non-maxima suppression.

    Returns
    -------
    model : `FFLDMixture`
        The newly trained model.
    """
    positive_image_arrays = []
    negative_image_arrays = []
    positive_bbox_arrays = []

    for image in positive_images:
        image_pixels = menpo_image_to_uint8(image)
        image_pixels = ensure_channel_axis(image_pixels)
        positive_image_arrays.append(image_pixels)
        im_bounding_boxes = []
        for lmark in image.landmarks.values():
            bb = lmark.bounding_box()
            height, width = bb.range()
            min_p, max_p = bb.bounds()
            im_bounding_boxes.append(np.array([min_p[1], min_p[0],
                                               width, height]))
        positive_bbox_arrays.append(im_bounding_boxes)

    for image in negative_images:
        image_pixels = menpo_image_to_uint8(image)
        image_pixels = ensure_channel_axis(image_pixels)
        negative_image_arrays.append(image_pixels)

    return train_model(positive_image_arrays, positive_bbox_arrays,
                       negative_image_arrays, n_components=n_components,
                       pad_x=pad_x, pad_y=pad_y, interval=interval,
                       n_relabel=n_relabel, n_datamine=n_datamine,
                       max_negatives=max_negatives, C=C, J=J, overlap=overlap)
