# DL_ASLR
Deep Learning CS 7643 Group Project

## Running the project locally

1. **Clone the Project Repository:**
   - Clone the repository using Git by running the following command:
     ```bash
     git clone https://github.com/Benler123/DL_ASLR.git
     ```
2. **Create and activate environment:**
    - (Skip if you have created the environment before). Run `conda env create -f environment.yml`. This will create an environment called `DL_ASLR_ENV`. 
    - Deactivate any current environments by running `conda deactivate`
    - Activate this environment by running `conda activate DL_ASLR_ENV`

3. **Preprocess Data**
    - Download the preprocessed data using [this link](https://www.dropbox.com/scl/fo/1tkb34i2xyjyl0xkdfmbj/ABjD3H-uKgYVivLEv8dMvjw?rlkey=t13vd2v643wjv8fy0vu0p6xo8&dl=0). 
    - Alternatively, you can download the original data  by running `python3 download_data.py` **locally** (~50 GB). Then, you can run `python3 preprocess_data.py` to generate the preprocessed data (~2GB). 
    - The preprocessed data should be named `X_train_combined` and `y_train_combined`. Put both these files in the `preprocessing` folder. 

3. **Modify Model and Experiment Names in `train_model.py`:**
   - Open `train_model.py` in a text editor of your choice.
   - Change MODEL_NAME to one of the following:
     - NN (Neural Network)
     - CNN (Convolutional Neural Network )
     - LSTM (Long Short-Term Memory )
   - Change EXPERIMENT_NAME to your desired experiment name

4. **Run the Training Script:**
    - Navigate to the root of the project `DL_ASLR`
   - In the terminal, execute the following command to start training:
     ```bash
     python3 train_model.py
     ```
   - The script will display training accuracy and loss per epoch as it runs.
   - Upon completion, it will print the final test accuracy and loss.
   - A graph depicting the training process will be generated and saved.
   - Adjust parameters and rerun as needed based on your experiment objectives.

## Running the project in PACE 
- Running the models locally can be time consuming due to limited RAM. If you have access to PACE, you can run this project in PACE to get faster results. 

### Log in to PACE

1. Open a terminal or command prompt.  
2. Log into PACE by running `ssh gburdell3@login-ice.pace.gatech.edu` (substitute your email). This will prompt you to log in using your GT password.

### One Time Conda Set Up in PACE 

- By default, conda creates new environments in the home directory at ~/.conda. However, conda environments can be quite large, and the allotted home directory has a relatively small storage quota.
  
1. First, check to see if you have an existing `~/.conda` directory or symlink by running:
    - `file ~/.conda`
2. If `file ~/.conda` reports "No such file or directory":
    1. Create a `.conda` directory in your scratch folder by running:
        - `mkdir ~/scratch/.conda`
    2. Then, create a symlink from your scratch folder to your home directory:
        - `ln -s ~/scratch/.conda ~/.conda`

3. If `file ~/.conda` reports that `~/.conda` is a directory, then:
    1. Move your `.conda` directory to your scratch space:
        - `mv ~/.conda ~/scratch/`
    2. Create a symlink from your project space to your home directory:
        - `ln -s ~/scratch/.conda ~/.conda`

### One Time Data Download

- Download the preprocessed data using [this link](https://www.dropbox.com/scl/fo/1tkb34i2xyjyl0xkdfmbj/ABjD3H-uKgYVivLEv8dMvjw?rlkey=t13vd2v643wjv8fy0vu0p6xo8&dl=0). 
- Alternatively, you can download the original data  by running `python3 download_data.py` **locally** (~50 GB). Then, you can run `python3 preprocess_data.py` to generate the preprocessed data (~2GB). 
- The preprocessed data should be named `X_train_combined` and `y_train_combined`. Your working directory should look like this: DL_ASLR, OnDemand, Scratch. Put the preprocessed data in the scratch directory. 

### Load Anaconda, allocate resources 

1. Load Anaconda by running:
    - `module load anaconda3/2022.05.0.1`

2. Use this command `salloc -N1 --mem-per-gpu=12G -t01:00:00 --gres=gpu:V100:1 --ntasks-per-node=4` to allocate yourself access to a GPU. In this case, it will allot a single node, a single V100 with 12gb of memory, 4 cores/threads for 1 hour. You can change the duration, nodes and type of GPU depending on your need, but this is a standard choice. 

3. You're now ready to run the project! Follow the steps in `Running the project locally` to clone the project and run the models. 

