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
        "pandas",
        "torch",
        "pytorch-lightning",
        "lmdb",
        "fastdtw",
        "tqdm",
        "pyro-ppl",
        "mltbox",
        "tyxe",
    ],
    python_requires=">=3.7",
    dependency_links=[
        "https://github.com/lbasora/mltbox/tarball/master#egg=mltbox",
        "https://github.com/lbasora/TyXe/tarball/master#egg=tyxe",
    ],
)
