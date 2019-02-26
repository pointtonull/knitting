from setuptools import setup, find_packages

setup(
    entry_points='''
        [console_scripts]
        knitting=src.main:cli
        iknitting=src.terminal:_main
    ''',
    include_package_data=True,
    install_requires=[open("requirements.txt").readlines()],
    name='knitting',
    packages=find_packages(),
    version='0.1',
)
