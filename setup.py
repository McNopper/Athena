"""
Setup configuration for the Athena package.

This setup file allows the project to be installed as a Python package,
making imports cleaner (e.g., `from src.model import LLM`).

Athena is named after the Greek goddess of knowledge, wisdom, and intelligence.
"""

from setuptools import setup, find_packages

setup(
    name="athena-llm",
    version="0.1.0",
    description="Athena - a modern LLM implementation for educational purposes",
    author="Norbert Nopper",
    url="https://github.com/McNopper/Athena",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
