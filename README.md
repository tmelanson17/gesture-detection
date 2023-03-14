# Motion History Image Project Folder

The report can be found [here](gatech_report.pdf).

- visualize_features.py : Compilation of various functions used for the report and for the presentation video.

- test_all.py : Main python file for running the various tests, as well as training and testing the final Python model.

- record_data.py : Attempt to record data into data files.  Converts the dataset into a series of pickle files

- labels.txt : List of labels for reference

### Non environment dependencies

matplotlib : For displaying the data

imageio : For fast image and video I/O

pickle : For storing temporary Python object data

### In src

dataset : Creates the data

dataviz : Helper functions for matplotlib displays

mhi : Contains the MHI class, which builds MHIs from sequences of images

extraction : Contains the extract_data method used to get the training and testing datasets

image_io : Contains the function for extracting data from a video, as well as writing out

humoments : used for calculating the Hu moments 

model : Contains the Trainer framework, which is a self-contained method to initialize a dataset, train a classifier, and evaluate it. Currenlty also supports multiple taus and majority filtering on the prediction.


## test : No functionality, only tests

The names should correspond to the modules the files test.

## images : Input folder

Contains the Action dataset in action_dataset.

## results : Output folder

train_data contains the relevant videos, written by write_mhi_output() in MHI in tests.



