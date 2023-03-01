from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'Image classification using Pre-trained Models'
LONG_DESCRIPTION = 'A package that allows to classifiy image dataset using pre-trained models and helps understand which model works best.'

# Setting up
setup(
    name="ptmodels",
    version=VERSION,
    author="MD Rafsun Sheikh",
    author_email="<201614123@student.mist.ac.bd>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'tensorflow', 'keras', 'scikit-learn'],
    keywords=['python', 'image', 'classification', 'pre-trained model', 'api', 'evaluation'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)