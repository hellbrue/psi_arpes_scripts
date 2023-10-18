import setuptools

setuptools.setup(
    name='fft_PSI',
    version='0.1',
    description='Filter the grid out of the EDC scans.',
    author='Francesco Barantani',
    packages=['fft_PSI'],
    install_requires=['numpy',' scipy'],
    keywords=[
     'ARPES',
    ],
)