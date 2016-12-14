from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

setup(
    name='pandas_vectors',
    version='0.1',
    description='convenience functions for dealing with vectors in panda dataframes',
    long_description=readme,
    author='Richard Joyce',
    author_email='rjoyce@ucdavis.edu',
    url='https://github.com/richjoyce/pandas_vectors',
    license='MIT',
    py_modules=['pandas_vectors']
)
