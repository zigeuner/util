from setuptools import setup
from setuptools import find_packages

from version import __version__

setup(
        name='util',
        packages=find_packages(),
        url='',
        license='',
        author='Lei Huang',
        author_email='lh389@cornell.edu',
        description='utilities package for infotopo',
        install_requires=['numpy',
                          'pandas',
                          'scipy',
                          'sympy',
                          'pprocess',
                          'matplotlib',
                          'seaborn',
                          'more_itertools',                          
                          ],  
        version=__version__,
        classifiers=['Topic :: Scientific/Engineering :: Bio-Informatics',
                     'Programming Language :: Python :: 2.7',
        ],
)
