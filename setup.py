from setuptools import setup, find_packages

setup(
    name="lolscan",
    version="0.1.0",
    description="YOLO model for detecting objects in League of Legends videos",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "lolscan=src.cli:main",
        ],
    },
)
