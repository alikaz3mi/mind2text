from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mind2text",
    version="0.1.0",
    author="Ali Kazemi",
    author_email="alikaz3mi@ieee.org",
    description="LLM-based Classification and Explainable Analysis of Motor Imagery EEG",
    long_description_content_type="text/markdown",
    url="https://github.com/alikaz3mi/mind2text",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0", "mypy>=0.910"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
)
