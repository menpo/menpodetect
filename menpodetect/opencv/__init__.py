from menpo.base import MenpoMissingDependencyError


try:
    from .detect import (load_opencv_frontal_face_detector,
                         load_opencv_profile_face_detector,
                         load_opencv_eye_detector, OpenCVDetector)
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError

