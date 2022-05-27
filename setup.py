import os
import site
import sys
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages

from setuptools import setup

# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Warn if we are installing over top of an existing installation. This can
# cause issues where files that were deleted from a more recent bbceas are
# still present in site-packages. See #18115.
overlay_warning = False
if "install" in sys.argv:
    lib_paths = [get_python_lib()]
    if lib_paths[0].startswith("/usr/lib/"):
        # We have to try also with an explicit prefix of /usr/local in order to
        # catch Debian's custom user site-packages directory.
        lib_paths.append(get_python_lib(prefix="/usr/local"))
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "bbceas"))
        if os.path.exists(existing_path):
            # We note the need for the warning here, but present it after the
            # command is run, so it's more likely to be seen.
            overlay_warning = True
            break

setup_info = dict(
    name="bbceas_processing",
    version="1.0.0",
    author="Net Lab",
    url="https://netlab.byu.edu/",
    download_url="https://github.com/NET-BYU/bbceas_data_analysis",
    description="Analysis and data collection for BBCEAS monitoring",
    license="BSD",
    packages=["bbceas_processing"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Add _ prefix to the names of temporary build dirs
    options={
        "build": {"build_base": "_build"},
    },
    zip_safe=True,
    install_requires=[
        "arrow==1.2.2",
        "numpy==1.22.3",
        "pandas==1.4.2",
        "scipy==1.8.0",
    ],
)


setup(**setup_info)


if overlay_warning:
    sys.stderr.write(
        """
========
WARNING!
========
You have just installed BBCEAS over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
BBCEAS. This is known to cause a variety of problems. You
should manually remove the
%(existing_path)s
directory and re-install BBCEAS.
"""
        % {"existing_path": existing_path}
    )
