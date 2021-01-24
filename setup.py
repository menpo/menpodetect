from setuptools import setup, find_packages


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("menpodetect")


setup(
    name="menpodetect",
    version=version,
    cmdclass=cmdclass,
    description="Object detection for Menpo",
    author="The Menpo Development Team",
    author_email="patricksnape@gmail.com",
    packages=find_packages(),
    package_data={"menpodetect": ["models/opencv/*.xml"]},
    install_requires=["menpo>=0.9.0,<0.12.0"],
    tests_require=["pytest>=5.0", "black>=20.0"],
)
