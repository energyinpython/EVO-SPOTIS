import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evo-spotis",
    version="0.0.6",
    author="Aleksandra Ba",
    author_email="aleksandra.baczkiewicz@phd.usz.edu.pl",
    description="Package for Multi-Criteria Decision Analysis with Preference Identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energyinpython/EVO-SPOTIS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
	install_requires=['numpy', 'scipy'],
)