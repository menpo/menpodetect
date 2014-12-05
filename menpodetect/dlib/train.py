import dlib

from menpodetect.detect import menpo_image_to_uint8
from .conversion import pointgraph_to_rect


def train_dlib_detector(images, output_path, epsilon=0.01,
                        add_left_right_image_flips=False, verbose_stdout=False,
                        C=5, detection_window_size=6400, num_threads=None):
    r"""
    Train a dlib detector with the given list of images.

    This is intended to easily train a list of menpo images that have their
    bounding boxes attached as landmarks. Each landmark group on the image
    will have a tight bounding box extracted from it and then dlib will
    train given these images. At the moment, the output is written to file
    and must be loaded back up when training is completed. No verbose
    messages will be provided by default.

    Parameters
    ----------
    images : list of menpo.image.Image
        The set of images to learn the detector from. Must have landmarks
        attached to **every** image, a bounding box will be extracted for each
        landmark group.
    output_path : Path or str
        The output path for dlib to save the detector to.
    epsilon : float, optional
        The stopping epsilon.  Smaller values make the trainer's solver more
        accurate but might take longer to train.
    add_left_right_image_flips : bool, optional
        If ``True``, assume the objects are left/right symmetric and add in
        left right flips of the training images.  This doubles the size of the
        training dataset.
    verbose_stdout : bool, optional
        If ``True``, will allow dlib to output its verbose messages. These
        will only be printed to the stdout, so will **not** appear in an IPython
        notebook.
    C : int, optional
        C is the usual SVM C regularization parameter.  Larger values of C will
        encourage the trainer to fit the data better but might lead to
        overfitting.
    detection_window_size : int, optional
        The number of pixels inside the sliding window used. The default
        parameter of 6400 = 80 * 80 window size.
    num_threads : int > 0 or None
        How many threads to use for training. If ``None``, will query
        multiprocessing for the number of cores.
    """
    rectangles = [[pointgraph_to_rect(lgroup.lms.bounding_box())
                   for lgroup in im.landmarks.values()]
                  for im in images]
    images = [menpo_image_to_uint8(im) for im in images]

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

    output_path_str = str(output_path)
    dlib.train_simple_object_detector(images, rectangles, output_path_str,
                                      options)
