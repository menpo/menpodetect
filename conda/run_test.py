import nose
import sys


if nose.run(argv=['', 'menpodetect']):
    sys.exit(0)
else:
    sys.exit(1)
