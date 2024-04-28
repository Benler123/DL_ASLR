# DL_ASLR
Deep Learning CS 7643 Group Project

## Staged Processed Data
[Link to Data](https://www.dropbox.com/scl/fo/1tkb34i2xyjyl0xkdfmbj/ABjD3H-uKgYVivLEv8dMvjw?rlkey=t13vd2v643wjv8fy0vu0p6xo8&dl=0)

## One Time Conda Set Up in PACE

- By default, conda creates new environments in the home directory at ~/.conda. However, conda environments can be quite large, and the allotted home directory has a relatively small storage quota.
  
1. Log into PACE by running `ssh gburdell3@login-ice.pace.gatech.edu` (substitute your email). This will prompt you to log in using your GT password.
2. First, check to see if you have an existing `~/.conda` directory or symlink by running:
    - `file ~/.conda`
3. If `file ~/.conda` reports "No such file or directory":
    1. Create a `.conda` directory in your scratch folder by running:
        - `mkdir ~/scratch/.conda`
    2. Then, create a symlink from your scratch folder to your home directory:
        - `ln -s ~/scratch/.conda ~/.conda`

4. If `file ~/.conda` reports that `~/.conda` is a directory, then:
    1. Move your `.conda` directory to your scratch space:
        - `mv ~/.conda ~/scratch/`
    2. Create a symlink from your project space to your home directory:
        - `ln -s ~/scratch/.conda ~/.conda`

## How to Use Environment in PACE

- Follow the one-time setup to create a conda symlink.
1. Log into PACE by running `ssh gburdell3@login-ice.pace.gatech.edu` (substitute your email). This will prompt you to log in using your GT password.
2. Go to the root of the project and run `conda env create -f environment.yml`.
3. This will create an environment called `DL_ASLR_ENV`. Activate this environment by running:
    - `conda activate DL_ASLR_ENV`
4. Load Anaconda by running:
    - `module load anaconda3/2022.05.0.1`
5. Deactivate any environment (if you are in base or any other). Activate your environment by running:
    - `activate DL_ASLR_ENV`

You're ready to run your project!
