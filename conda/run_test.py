import nose
import sys
import os
import menpodetect


is_py3k = sys.version_info.major == 3
menpodetect_path = os.path.dirname(menpodetect.__file__)
nose_args = ['', 'menpodetect']


# OpenCV 2.x does not support Python 3
if is_py3k:
    nose_args += ['--exclude-dir={}'.format(os.path.join(menpodetect_path,
                                                         'opencv'))]

if nose.run(argv=nose_args):
    sys.exit(0)
else:
    sys.exit(1)
