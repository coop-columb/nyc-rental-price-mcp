from setuptools import setup, find_packages
import os

# Read version from __init__.py or use default
version_file = os.path.join("src", "nyc_rental_price", "__init__.py")
version = "0.1.0"
try:
    with open(version_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.startswith("__version__")]
        if lines:
            version = lines[0].split("=")[1].strip().strip('"\'')
except (FileNotFoundError, IOError):
    pass

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except (FileNotFoundError, IOError):
    long_description = "NYC Rental Price Prediction Project"

setup(
    name="nyc_rental_price",
    version=version,
    author="NYC Rental Price Prediction Team",
    author_email="example@example.com",
    description="Machine learning model for predicting NYC rental prices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nyc-rental-price-mcp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "requests>=2.26.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=22.3.0",
            "ruff>=0.0.262",
            "mypy>=0.991",
            "isort>=5.12.0",
            "pre-commit>=3.2.0",
            "types-requests>=2.28.0",
            "types-PyYAML>=6.0.0",
            "coverage>=7.2.0",
            "interrogate>=1.5.0",
            "nbstripout>=0.6.1",
            "nbqa>=1.7.0",
        ],
        "doc": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.18.0",
        ],
        "perf": [
            "pytest-benchmark>=4.0.0",
            "memory-profiler>=0.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nyc-train=nyc_rental_price.models.train:main",
            "nyc-predict=nyc_rental_price.models.predict:main",
            "nyc-api=nyc_rental_price.api.main:run_api",
        ],
    },
    include_package_data=True,
    keywords="nyc, rental, price, prediction, machine learning, pytorch, tensorflow",
    zip_safe=False,
)