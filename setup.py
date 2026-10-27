from setuptools import find_packages, setup

setup(
    name="nekabl",
    version="0.5.0",
    description="Exploratory analysis of atmospheric boundary layer LES from Neko and Nek5000",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="L. L. Donati",
    packages=find_packages(exclude=["examples"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "xarray>=2022.6", "pandas>=1.4", "matplotlib>=3.5"],
    entry_points={"console_scripts": ["nekabl-summary=nekabl.summary:main",
                                      "nekabl-explore=nekabl.explore:main"]},
)
