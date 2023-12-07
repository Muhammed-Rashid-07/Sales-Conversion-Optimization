"""The setup script. It uses setuptools to configure your package with metadata 
(name, version, author, etc.) and dependencies."""

from setuptools import find_packages,setup
from typing_extensions import List

HYPHEN_DOT = '-e .'
def get_requirements(filepath:str) -> List[str]:
    """
    Returns the list of requirements from the requirements.txt file
    """
    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
    return requirements

setup(
    name='sales_conversion_optimization_prediction',
    version='0.0.1',    
    description='Sales Conversion Optimization Prediction',
    author='Rashid',
    author_email='rashid24601@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    

)