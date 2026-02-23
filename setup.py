"""Setup script for ECCV framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "REFACTORED_README.md").read_text(encoding='utf-8')

setup(
    name="eccv-framework",
    version="1.0.0",
    author="Prakarn Taranodom",
    author_email="your.email@example.com",
    description="Enhanced Clustering using Conditional Volatility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Prakarn-Taranodom/eccv-clustering-framework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aeon>=0.8.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "scipy>=1.7.0",
        "arch>=5.0.0",
        "tslearn>=0.5.2",
        "pmdarima>=2.0.0",
        "scikit-learn-extra>=0.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "nbstripout>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eccv-benchmark=examples.02_batch_benchmark:main",
        ],
    },
)
