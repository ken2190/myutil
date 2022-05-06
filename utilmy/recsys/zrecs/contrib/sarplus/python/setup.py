import sysconfig

from setuptools import setup
from setuptools.extension import Extension


class get_pybind_include(object):
    def __init__(self, user=False):
        """ get_pybind_include:__init__.
        Doc::
                
                    Args:
                        user:     
                    Returns:
                       
        """
        self.user = user

    def __str__(self):
        """ get_pybind_include:__str__.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        import pybind11

        return pybind11.get_include(self.user)


setup(
    name="pysarplus",
    version="0.2.6",
    description="SAR prediction for use with PySpark",
    url="https://github.com/Microsoft/Recommenders/contrib/sarplus",
    author="Markus Cozowicz",
    author_email="marcozo@microsoft.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["pytest-runner"],
    install_requires=["pybind11>=2.2"],
    tests_require=["pytest"],
    packages=["pysarplus"],
    ext_modules=[
        Extension(
            "pysarplus_cpp",
            ["src/pysarplus.cpp"],
            include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
            extra_compile_args=sysconfig.get_config_var("CFLAGS").split()
            + ["-std=c++11", "-Wall", "-Wextra"],
            libraries=["stdc++"],
            language="c++11",
        )
    ],
    zip_safe=False,
)
