from menpo.base import MenpoMissingDependencyError


try:
    from .detect import load_bob_frontal_face_detector, BobDetector
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
