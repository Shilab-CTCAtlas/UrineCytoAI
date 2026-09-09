import os
from setuptools import setup, find_packages

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as fh:
            return fh.read()
    return "Cross-domain cluster matching pipeline for cell image analysis."

with open("requirements.txt", "r", encoding="utf-8") as handle:
    requirements = [
        line.strip() for line in handle
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="shilab-cluster-algorithm",
    version="1.0.0",
    author="QRR",
    description="A cross-domain cluster matching pipeline for analyzing cell images",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
