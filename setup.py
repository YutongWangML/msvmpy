from setuptools import setup

setup(
    name="msvmpy",
    version="0.1.0",
    author="Yutong Wang",
    description="Multiclass SVM",
    packages=["msvmpy"],
    install_requires=[
        "cvxopt",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
