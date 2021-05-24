# Always prefer setuptools over distutils
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / Path('README.md')) as f:
    long_description = f.read()

setup(
    name='pytisch',
    version='0.1.0',
    description='OCR library for tabular data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hadware/pytisch',
    author='Hadrien Titeux',
    author_email='hadrien.titeux@ens.fr',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),
    setup_requires=['setuptools>=38.6.0'],  # >38.6.0 needed for markdown README.md
    install_requires=[
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "Pillow",
        "scikit-learn",
        "typing-extensions"
    ],
    extras_requires={
        "tesseract": [
            "pytesseract",
        ],
        "easyocr": [
            "easyocr",
        ],
        "testing": {
            "pytest",
            "easyocr",
            "pytesseract"
        }
    }
)
