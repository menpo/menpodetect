from menpo.base import MenpoMissingDependencyError


try:
    from .detect import FFLD2Detector, load_ffld2_frontal_face_detector
    from .train import train_ffld2_detector
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
