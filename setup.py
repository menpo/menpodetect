from setuptools import setup, find_packages
import versioneer

project_name = 'menpodetect'

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.
versioneer.VCS = 'git'
versioneer.versionfile_source = '{}/_version.py'.format(project_name)
versioneer.versionfile_build = '{}/_version.py'.format(project_name)
versioneer.tag_prefix = 'v'  # tags are like v1.2.0
versioneer.parentdir_prefix = project_name + '-'  # dirname like 'menpo-v1.2.0'


# Also requires the dlib package
requirements = ['numpy>=1.9,<=1.10',
                'cypico==0.2.2',
                'menpo==0.4.0a3']


setup(name=project_name,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Object detection for Menpo',
      author='The Menpo Development Team',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      tests_require=['nose==1.3.4'],
      package_data={'menpodetect': ['models/opencv/*.xml']},
      install_requires=requirements)
