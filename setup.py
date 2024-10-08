from setuptools import setup, find_packages

setup(
    name='thesis_project',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'river',
        'evidently',
        'matplotlib',
        'seaborn',
    ],
)
