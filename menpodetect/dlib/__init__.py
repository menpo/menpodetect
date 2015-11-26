from menpo.base import MenpoMissingDependencyError


try:
    from .detect import load_dlib_frontal_face_detector, DlibDetector
    from .train import train_dlib_detector
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
