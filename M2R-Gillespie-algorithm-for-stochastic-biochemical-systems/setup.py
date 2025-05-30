from setuptools import setup, find_packages # noqa

setup(
    name="M2R-Gillespie-Algorithm",
    version="0.1.0",
    description="Gillespie SSA implementations for stochastic biochemical systems", # noqa
    author="Yanbo Huang",
    author_email="yh2723@ic.ac.uk",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
