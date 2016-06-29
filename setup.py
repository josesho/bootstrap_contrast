from setuptools import setup, find_packages

setup(name='bootstrapContrast',
      version='0.227',
      description='Calculation and Visualization of Confidence Intervals and Effect Sizes for Python.',
      packages = find_packages(),
      install_requires = ['numpy','scipy','pandas','seaborn','matplotlib'],
      url='http://github.com/josesho/bootstrapContrast',
      author='Joses Ho',
      author_email='joseshowh@gmail.com',
      license='GNU GPLv3',
      zip_safe=False)
