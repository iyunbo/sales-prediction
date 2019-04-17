from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas>=0.23.0', 'sklearn', 'xgboost>=0.81', 'numpy', 'matplotlib', 'joblib', 'sqlalchemy']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Rossmann training application package.'
)
