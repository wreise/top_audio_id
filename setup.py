from setuptools import setup

setup(
    name="audioId",
    version="0.1.0",
    author="Wojciech Reise",
    author_email="reisewojciech@gmail.com",
    packages=["audioId"],
    scripts=[],
    url="",
    license="",
    description="Package with utils for audio manipulation and identification",
    install_requires=[
        "librosa==0.9.2",
        "gudhi==3.7.1",
        "matplotlib",
        "scipy",
        "pytest",
        "jupyter",
        "tqdm",
        "opencv-python",
        "sox",
        "termcolor",
        "pydub",
    ],
)
