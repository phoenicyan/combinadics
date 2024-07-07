import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="combinadics",
    version="0.0.1",
    author="Konstantin Izmailov",
    author_email="kizmailov@gmail.com",
    description="A fast combinations calculation library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: FSF Approved :: GNU3",
        "Operating System :: Linux",
    ],
    python_requires='>=3.9',
    install_requires=requirements,
)
