from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read().strip()


def version():
    with open("Hy2DL/__version__.py") as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc["__version__"]


def requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")


setup(name="hy2dl",
      license="GPLv3",
      version=version(),
      author="",
      author_email="",
      description="",
      long_description=readme(),
      long_description_content_type="text/markdown",
      install_requires=requirements(),
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      )
