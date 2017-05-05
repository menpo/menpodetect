from setuptools import setup, find_packages
import versioneer


setup(name='menpodetect',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Object detection for Menpo',
      author='The Menpo Development Team',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      tests_require=['nose'],
      package_data={'menpodetect': ['models/opencv/*.xml']},
      install_requires=['menpo>=0.8,<0.9'])

