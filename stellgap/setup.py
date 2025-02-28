from setuptools import setup, find_packages

setup(
    name='stellgap',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'plotly'
    ],
    author='Alex Knayzev',
    author_email='a.knyazev@columbia.edu',
    description='A package to work with STELLGAP output',
    keywords='physics plasma alfven continuum',
)
