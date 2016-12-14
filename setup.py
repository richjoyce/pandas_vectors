from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pandas_vectors',
    version='0.0.1',
    description='convenience functions for dealing with vectors in panda dataframes',
    long_description=readme,
    author='Richard Joyce',
    author_email='rjoyce@ucdavis.edu',
    url='https://github.com/richjoyce/pandas_vectors',
    license=license,
    packages=['pandas_vectors']
)
