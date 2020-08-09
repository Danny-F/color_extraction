from setuptools import setup, find_packages

setup(
   name='color_extraction',
   version='1.0',
   description='Runs an interactive streamlit instance, that will extract colors \
   from and image and display them in a pie graph.',
   author='Danny Farrington',
   author_email='dannyfarrington5@gmail.com',
   packages=['color_extraction'],
   install_requires=['numpy', 'pandas', 'matplotlib',
                     'sklearn', 'scikit-image', 'streamlit'],
)
