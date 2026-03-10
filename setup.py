from setuptools import setup,find_packages
from typing import List

Hyphen_dot='-e.'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if Hyphen_dot in requirements:
            requirements.remove(Hyphen_dot)
    return requirements
setup(
    name="ML_ASP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )