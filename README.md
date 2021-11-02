# VAD_assignment

This repo implements a VAD using two stages: 1) spatial module for speech enhancement 2) a DNN-based predictive model. 
    
## Instructions

Clone the repository then follow these instructions. Global note: You would have to adjust the directories within each file in the code according to your machine. 

1. To build the docker image run:
```
docker build -t vad_image docker
```

2. Run the container with: 
```
nvidia-docker run -ti --rm --memory=20g --name=vad_container -p 8888:8888 vad_image
```
Note: you might need to adjust the memory or port according to your machine. 

3. Download the datasets and the repo inside the docker

4. For applying the spatial modules, use the spatial_processing.ipynb script. This will generate all the output within the same parent directory, but each output is inside a directory named after the module. 

5. To prepare the dataset for training and testing the DNN-model, follow the data_preprocessing.ipynb. (Note: this also includes some data inspection and visualization). This will output the melspectrograms and its frame-level labels. These outputs will be generated in directories names `[spatialModule]_mels_labels'

6. Finally, follow the code in cnn_vad.ipynb or crnn_vad.ipynb to train and test the DNN models. 
 
**Note:** To run the jupyter notebook within docker use 
```
jupyter notebook --allow-root --ip=$(awk 'END{print $1}' /etc/hosts) --no-browser --NotebookApp.token= &
```
Then access the notebook through the designated port.

## Items 
This repository contains the following item: 
- '**data_preprocessing.ipynb:**' This is a guided notebook that includes the first data inspection, then the preprocessing required to prepare the dataset for training and testing the DNN models. Note: this is to be applied after the spatial processing stage.
- '**spatial_processing.ipynb**' This notebook contains the implementation for the spatial processing modules. It also includes samples of the output of each module. 
- '**cnn_vad.ipynb**' This notebook contains the implementation for the CNN-based models. It also contains the code for training and testing the model using all possible spatial modules. 
- '**crnn_vad.ipynb**' Similarly, this notebook contains the implementation for the CRNN-based models.
- '**playground.ipynb**' This notebook shows some code I was trying to get familiar with pyroomacoustics. 
- '**vad_end_to_end.py**' his file contains the full code to run both the spatial modules and the DNN-based model on an input file. The chosen model and spatial modules can be passed along with the target file and its metadata. Note: this requires having the pretrained models already available. (Not tested yet)
- '**requirements.txt**' contains the required packages to run the code. Only needed in case docker is not used.   
- '**dockerfile**' This file can be used to build and run a docker container with the required libraries pre-installed. 

The repository contains one directories: 
- '**Docker**' contains the script for building the docker image and installing the requirements.
