import dlib

from menpodetect.detect import menpo_image_to_uint8
from .conversion import pointgraph_to_rect


def train_dlib_detector(images, output_path, epsilon=0.01,
                        add_left_right_image_flips=False, verbose_stdout=False,
                        C=5, detection_window_size=6400, num_threads=None):
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
