from menpo.base import MenpoMissingDependencyError

try:
    import dlib
except ImportError:
    raise MenpoMissingDependencyError('dlib')

from menpodetect.detect import menpo_image_to_uint8
from .conversion import pointgraph_to_rect


def train_dlib_detector(images, epsilon=0.01, add_left_right_image_flips=False,
                        verbose_stdout=False, C=5, detection_window_size=6400,
                        num_threads=None):
    r"""
    Train a dlib detector with the given list of images.

    This is intended to easily train a list of menpo images that have their
    bounding boxes attached as landmarks. Each landmark group on the image
    will have a tight bounding box extracted from it and then dlib will
    train given these images.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images to learn the detector from. Must have landmarks
        attached to **every** image, a bounding box will be extracted for each
        landmark group.
    epsilon : `float`, optional
        The stopping epsilon.  Smaller values make the trainer's solver more
        accurate but might take longer to train.
    add_left_right_image_flips : `bool`, optional
        If ``True``, assume the objects are left/right symmetric and add in
        left right flips of the training images.  This doubles the size of the
        training dataset.
    verbose_stdout : `bool`, optional
        If ``True``, will allow dlib to output its verbose messages. These
        will only be printed to the stdout, so will **not** appear in an IPython
        notebook.
    C : `int`, optional
        C is the usual SVM C regularization parameter.  Larger values of C will
        encourage the trainer to fit the data better but might lead to
        overfitting.
    detection_window_size : `int`, optional
        The number of pixels inside the sliding window used. The default
        parameter of ``6400 = 80 * 80`` window size.
    num_threads : `int` > 0 or ``None``
        How many threads to use for training. If ``None``, will query
        multiprocessing for the number of cores.

    Returns
    -------
    detector : `dlib.simple_object_detector`
        The trained detector. To save this detector, call save on the returned
        object and pass a string path.

    Examples
    --------
    Training a simple object detector from a list of menpo images and save it
    for later use:

    >>> images = list(mio.import_images('./images/path'))
    >>> in_memory_detector = train_dlib_detector(images, verbose_stdout=True)
    >>> in_memory_detector.save('in_memory_detector.svm')
    """
    rectangles = [[pointgraph_to_rect(lgroup.bounding_box())
                  for lgroup in im.landmarks.values()]
                  for im in images]
    image_pixels = [menpo_image_to_uint8(im) for im in images]

    if num_threads is None:
        import multiprocessing

        num_threads = multiprocessing.cpu_count()

    options = dlib.simple_object_detector_training_options()
    options.epsilon = epsilon
    options.add_left_right_image_flips = add_left_right_image_flips
    options.be_verbose = verbose_stdout
    options.C = C
    options.detection_window_size = detection_window_size
    options.num_threads = num_threads

    return dlib.train_simple_object_detector(image_pixels, rectangles, options)
