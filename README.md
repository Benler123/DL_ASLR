# DL_ASLR
Deep Learning CS 7643 Group Project

## Staged Processed Data ##
https://www.dropbox.com/scl/fo/1tkb34i2xyjyl0xkdfmbj/ABjD3H-uKgYVivLEv8dMvjw?rlkey=t13vd2v643wjv8fy0vu0p6xo8&dl=0 

## One Time Conda Set Up in PACE ##

By default, conda creates new environments in the home directory at ~/.conda. However, conda environments can be quite large, and the alloted home directory has a relatively small storage quota.

To resolve this, we can create a symlink named ~/.conda that points to the scratch directory, which has a much larger storage quota. 

ONE TIME SET UP: 
0. Log into pace by running `ssh gburdell3@login-ice.pace.gatech.edu` (substitute your email). This will prompt you to log in using your GT password. 
1. First, check to see if you have an existing ~/.conda directory or symlink by running: `file ~/.conda`. 
2. If file ~/.conda reports "No such file or directory": 
2.1. Create a .conda directory in your scratch folder by running `mkdir ~/scratch/.conda`. 
2.2 Then, create a symlink from your scratch folder to your home directory:
`ln -s ~/scratch/.conda ~/.conda`. 

3. If file ~/.conda reports that ~/.conda is a directory, then:
3.1 Move your .conda directory to your scratch space. 
mv ~/.conda ~/scratch/
3.2 Create a symlink from your project space to your home directory:
ln -s ~/scratch/.conda ~/.conda

## How to use environment in PACE ##

0. Follow the one time set up to create a conda sym link. 
1. Log into pace by running `ssh gburdell3@login-ice.pace.gatech.edu` (substitute your email). This will prompt you to log in using your GT password. 
2. Go to the root of the project and run `conda env create -f environment.yml`
3. This will create an environment called `DL_ASLR_ENV`. Activate this environment by running conda activate `DL_ASLR_ENV`
4. Load anaconda by running `module load anaconda3/2022.05.0.1`
3. Deactivate any environment (if you are in base or any other). Activate your environment by running `activate DL_ASLR_ENV`. You're ready to run your project!