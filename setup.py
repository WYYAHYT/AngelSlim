"""Setup for pip package."""
import os
import subprocess

from setuptools import find_packages, setup

# 获取 setup.py 所在目录的绝对路径
here = os.path.abspath(os.path.dirname(__file__))

TOOLS_VERSION = None

if "main" in subprocess.getoutput("git branch"):
    TOOLS_VERSION = "0.0.0_dev"
else:
    tag_list = subprocess.getoutput("git tag").split("\n")
    TOOLS_VERSION = tag_list[-1]


def get_requirements(filename):
    """Load dependency packages from specified requirements file"""
    # 使用绝对路径，基于 setup.py 所在目录
    req_path = os.path.join(here, filename)
    with open(req_path, encoding='utf-8') as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith(("#", "-"))
        ]


setup(
    name="angelslim",
    version=TOOLS_VERSION,
    description=("A toolkit for compress llm model."),
    long_description="Tools for llm model compression",
    url="https://github.com/Tencent/AngelSlim",
    author="Tencent Author",
    # Core dependencies: installed by default
    install_requires=get_requirements("requirements/requirements.txt"),
    # Define optional dependency groups
    extras_require={
        # Install all optional features: pip install angelslim[all]
        "all": (
            get_requirements("requirements/requirements_speculative.txt")
            + get_requirements("requirements/requirements_diffusion.txt")
            + get_requirements("requirements/requirements_multimodal.txt")
            + get_requirements("requirements/requirements_benchmark.txt")
        ),
        # Install speculative sampling functionality: pip install angelslim[speculative]
        "speculative": get_requirements("requirements/requirements_speculative.txt"),
        # Install Diffusion functionality: pip install angelslim[diffusion]
        "diffusion": get_requirements("requirements/requirements_diffusion.txt"),
        # Install multimodal functionality: pip install angelslim[multimodal]
        "multimodal": get_requirements("requirements/requirements_multimodal.txt"),
        # Install benchmark functionality: pip install angelslim[benchmark]
        "benchmark": get_requirements("requirements/requirements_benchmark.txt"),
    },
    packages=find_packages(),
    python_requires=">=3.0",
    # PyPI package information.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="License for AngelSlim",
    keywords=("Tencent large language model model-optimize compression toolkit."),
)
