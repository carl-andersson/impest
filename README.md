# Impulse response estimation

The implementation is made in Tensorflow for python and has been tested on Tensorflow 1.12 and python3.6. Other required packages are sklearn and numpy. No data is included to keep the repo small but can be generated using generated_data.m

##First setup 
1. Install tensorflow and dependecies
2. Run the following to generate data in matlab
    1. `generate_data("data_train.mat",10000)`
    2. `generate_data("data_test.mat",1000)`
3. Start training `pyhton main.py`

