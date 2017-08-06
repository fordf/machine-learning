from setuptools import setup

setup(
    name="machine-learning",
    description="Vanilla/numpy implementations of a few machine-learning tools",
    version=0.2,
    author=["Ford Fowler", "Claire Gatenby"],
    author_email=["fordjfowler@gmail.com",
                  'clairejgatenby@gmail.com'],
    licencse="MIT",
    packages=['ml_algs'],
    package_dir={'ml_algs': 'ml_algs'},
    install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib'],
    extras_require={
        "test": ["pytest", "pytest-cov", "tox"]
    },
)
