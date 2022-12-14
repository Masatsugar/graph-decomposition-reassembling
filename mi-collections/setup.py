import setuptools
from mi_collections import __version__

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="mi_collections",  # Replace with your own username
    version=__version__,
    entry_points={
        "console_scripts": [
            "corona=corona:main",
        ],
    },
    author="Masatsugar",
    author_email="example@example.com",
    description="A MI packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Masatsugar/mi_collections",
    packages=setuptools.find_packages(),
    install_requires=[
        # "numpy",
        "networkx",
        "grakel",
        "tqdm",
        "joblib",
        "pandas",
        # "matplotlib",
        # "IPython",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
