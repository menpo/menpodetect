"""What menpodetect says when OpenCV or dlib cannot be imported. Each case runs in a fresh interpreter in which the
library is blocked, so that it does not matter whether it is installed where the tests run."""

import subprocess
import sys

import pytest

from menpo.base import MenpoMissingDependencyError
from menpodetect._optional import MissingExtraError, names_of_a_missing_library

BLOCK = "import sys; sys.modules[{!r}] = None\n"
# a library that is there but fails to load, as OpenCV does without libGL
BREAK = """
import sys
class Breaks:
    def find_spec(self, name, path=None, target=None):
        if name == {!r}:
            raise ImportError("libGL.so.1: cannot open shared object file")
sys.meta_path.insert(0, Breaks())
"""


def run(prelude, statement):
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", prelude + statement],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize(
    "statement",
    [
        "from menpodetect.opencv import load_opencv_frontal_face_detector",
        "from menpodetect.opencv import load_opencv_profile_face_detector",
        "from menpodetect.opencv import load_opencv_eye_detector",
        "from menpodetect.opencv import OpenCVDetector",
        "import menpodetect.opencv as o; o.OpenCVDetector",
        "import menpodetect.opencv.detect",
    ],
)
def test_opencv_missing_says_to_install_the_extra(statement):
    code, out, stderr = run(BLOCK.format("cv2"), statement)
    assert code != 0 and "MissingExtraError" in stderr
    assert 'pip install "menpodetect[opencv]"' in stderr
    assert (
        "The import failed with" in stderr and "ModuleNotFoundError" in stderr
    )  # the real error
    assert "conda" not in stderr and "pip install opencv\n" not in stderr


@pytest.mark.parametrize(
    "statement",
    [
        "from menpodetect.dlib import load_dlib_frontal_face_detector",
        "from menpodetect.dlib import DlibDetector",
        "from menpodetect.dlib import train_dlib_detector",
        "import menpodetect.dlib.detect",
        "import menpodetect.dlib.train",
    ],
)
def test_dlib_missing_says_to_install_the_extra(statement):
    code, out, stderr = run(BLOCK.format("dlib"), statement)
    assert code != 0 and 'pip install "menpodetect[dlib]"' in stderr
    assert "The import failed with" in stderr and "ModuleNotFoundError" in stderr


def test_the_missing_library_error_is_an_import_error_that_can_be_caught():
    code, out, stderr = run(
        BLOCK.format("cv2"),
        "try:\n"
        "    from menpodetect.opencv import OpenCVDetector\n"
        "except ImportError as e:\n"
        "    print(type(e).__name__, 'caught')\n",
    )
    assert code == 0, stderr
    assert "MissingExtraError caught" in out


def test_other_names_of_the_package_are_still_attribute_errors():
    code, out, stderr = run(
        BLOCK.format("cv2"),
        "import menpodetect.opencv as o\n"
        "print(hasattr(o, 'not_a_name'))\n"
        "try:\n    o.not_a_name\nexcept AttributeError as e:\n    print(e)\n",
    )
    assert code == 0, stderr
    assert out.splitlines() == [
        "False",
        "module 'menpodetect.opencv' has no attribute 'not_a_name'",
    ]


def test_importing_the_package_itself_still_works_without_the_libraries():
    # menpodetect/__init__ does "from menpodetect.opencv import *": nothing is there, and nothing fails
    code, out, stderr = run(
        BLOCK.format("cv2") + BLOCK.format("dlib"),
        "import menpodetect, menpodetect.opencv, menpodetect.dlib\n"
        "print(hasattr(menpodetect, 'OpenCVDetector'), hasattr(menpodetect, 'DlibDetector'))\n",
    )
    assert code == 0, stderr
    assert out.strip() == "False False"


def test_a_library_that_is_installed_but_fails_to_load_is_not_called_missing():
    code, out, stderr = run(
        BREAK.format("cv2"), "from menpodetect.opencv import OpenCVDetector"
    )
    assert code != 0
    assert "could not be imported" in stderr and "libGL.so.1" in stderr
    assert "Install it with" not in stderr and "pip install" not in stderr


def test_missing_extra_error_for_a_library_that_is_not_installed():
    error = MissingExtraError(
        "opencv",
        "OpenCV",
        "cv2",
        ModuleNotFoundError("No module named 'cv2'", name="cv2"),
    )
    assert isinstance(
        error, MenpoMissingDependencyError
    )  # what menpodetect's __init__ catches
    assert error.missing_name == "opencv" and error.extra == "opencv"
    assert 'pip install "menpodetect[opencv]"' in str(error)


def test_missing_extra_error_without_an_error_assumes_it_is_not_installed():
    assert 'pip install "menpodetect[dlib]"' in str(
        MissingExtraError("dlib", "dlib", "dlib")
    )


@pytest.mark.parametrize(
    "error",
    [
        ImportError("libGL.so.1: cannot open shared object file"),
        ImportError("DLL load failed"),
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),  # another module
    ],
)
def test_missing_extra_error_for_a_failure_that_is_not_the_library_missing(error):
    result = MissingExtraError("opencv", "OpenCV", "cv2", error)
    assert result.missing_name is None
    assert "could not be imported" in str(result) and "pip install" not in str(result)
    assert str(error) in str(result)


def test_names_of_a_missing_library_raises_the_error_for_its_names_only():
    error = MissingExtraError("opencv", "OpenCV", "cv2")
    getattr_ = names_of_a_missing_library(error, ("a", "b"), "pkg")
    with pytest.raises(MissingExtraError):
        getattr_("a")
    with pytest.raises(AttributeError, match="module 'pkg' has no attribute 'c'"):
        getattr_("c")
