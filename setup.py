from setuptools import setup

setup(
    name="sconegym",
    version="0.2.0",
    description="SconePy Gym Environments",
    url="https://scone.software/",
    license="Apache 2.0",
    packages=["sconegym"],
    package_data={"sconegym": ["data/*.*", "data-v1/*.*"]},
    include_package_data=True,
    python_requires="~=3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["gym==0.13"],
)
