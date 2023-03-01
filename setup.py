import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='ptmodels',  
     version='0.1',
     scripts=['setup'] ,
     author="MD Rafsun Sheikh",
     author_email="201614123@student.mist.ac.bd",
     description="ptmodels uses pre-trained models to evaluate image datasets and helps to understand which model works",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/rafsunsheikh/ptmodels",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.1",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )