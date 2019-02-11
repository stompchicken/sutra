from setuptools import setup, find_packages
from os import path

root_dir = path.abspath(path.dirname(__file__))

with open(path.join(root_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sutra',
    version='0.0.1',
    description='Natural language processing in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stompchicken/sutra',
    author='Stephen Spencer',
    author_email='sutra@stompchicken.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['sutra', 'tests']),
    install_requires=['torch', 'numpy', 'pandas', 'requests', 'psutil']
)
