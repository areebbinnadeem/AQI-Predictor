from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != HYPHEN_E_DOT]

    return requirements

setup(
    name='aqi-predictor',
    version='0.0.1',
    author='Areeb',
    author_email='M.AreebBinNadeem@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
