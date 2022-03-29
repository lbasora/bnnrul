from setuptools import setup, find_packages

setup(
    name="bnnrul",
    version="0.1",
    description="Bayesian Neural Networks for Remaining Useful Life estimation",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "pandas",
        "pytorch-lightning",
        "lmdb",
        "fastdtw",
        "tqdm",
        "h5py",
        "nb_black",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
