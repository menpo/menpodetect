from menpo.base import MenpoMissingDependencyError


try:
    from .detect import load_pico_frontal_face_detector, PicoDetector
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
