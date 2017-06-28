from setuptools import setup, find_packages

setup(name='bootstrap_contrast',
      version='0.311',
      description='Calculation and Visualization of Confidence Intervals and Effect Sizes for Python.',
      packages = find_packages(),
      install_requires = ['numpy','scipy','pandas','seaborn','matplotlib'],
      url='http://github.com/josesho/bootstrap_contrast',
      author='Joses Ho',
      author_email='joseshowh@gmail.com',
      license='GNU GPLv3',
      zip_safe=False)
