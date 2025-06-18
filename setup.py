from setuptools import find_packages,setup
from typing import List

Hypen_e_dot='-e .'
def get_requirments(file_path:str)->list(str):
    ''' 
    This Function will return the list of Requirments 
    '''

    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        [req.replace("/n","") for req in requirments]

        if Hypen_e_dot in requirments:
            requirments.remove(Hypen_e_dot)

    return requirments

setup(
    name='TelCom Fraud Detection',
    version ='0.0.1',
    author='Minega Raju',
    author_email='raju.meenige@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt')


)
