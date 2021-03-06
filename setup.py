from setuptools import setup, find_packages
import os
# Taken from setup.py in seaborn.
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"]="."

# Modified from from setup.py in seaborn.
try:
    from setuptools import setup
    _has_setuptools=True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    to_install=[]

    try:
        import numpy
    except ImportError:
        to_install.append('numpy>=1.13.1')
    try:
        import scipy
    except ImportError:
        to_install.append('scipy>=0.19.1')
    try:
        import matplotlib
    except ImportError:
        to_install.append('matplotlib>=2.0.2')
    try:
        import pandas
        if int(pandas.__version__.split('.')[1])<20:
            to_install.append('pandas>=0.20.1')
    except ImportError:
        to_install.append('pandas>=0.20.1')
    try:
        import seaborn
    except ImportError:
        to_install.append('seaborn>0.8')

    return to_install

if __name__=="__main__":

    installs=check_dependencies()
    setup(name='bootstrap_contrast',
    author='Joses Ho',
    author_email='joseshowh@gmail.com',
    version=1.0,
    description='Calculation and Visualization of Confidence Intervals and Effect Sizes for Python.',
    packages=find_packages(),
    install_requires=installs,
    url='http://github.com/josesho/bootstrap_contrast',
    license='MIT'
    )
