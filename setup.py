from setuptools import setup, find_packages

exec(open('fast/__version__.py').read())
setup(
    name="fast",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="NA",
    license="MIT",
    python_requires='>=3.7', 
    entry_points={'console_scripts': [
        # train fast
        'train_fast=fast.train_fast:main',
        ]
        }
)
