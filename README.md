# Satellite Image Classifier for Determining Natural Resource Richness

Our team will be creating a binary classification model for satellite images of the Earth. Our objective is
to train an agent to classify landsat imaging on whether the area is suitable for resource extraction.  

NASA's Landsat program has created a publicly available dataset of Earth observation images over many
years, labeled according to date and GPS coordinates. The dataset is an extensive library of images that
we will have to cross reference against existing resource deposits. We will develop an algorithm to
preprocess the data, making reference to a supplementary dataset containing the GPS coordinates of
known mineral / metal deposits. The preprocessor will convert the GPS information associated with a
Landsat image into a label describing the available resources, or lack thereof. We will use this method to
generate a sizable dataset to be used for training and validation. We have identified several
complementary natural resource datasets which can be used for preprocessing.  

In the unlikely case that the above preprocessing is not possible, or that the problem proves intractable,
we will instead attempt to classify Landsat images according to another label scheme. One possibility is to
predict the likelihood that an image is near a seismically active region using geospatially labeled
earthquake data. Or, as an exercise, we will attempt to predict the province, state, or country that is
depicted in an image. Predicting world location based on satellite imagery may not have practical
applications, but it is still an interesting ML problem and may pose interesting design challenges.

After preparing the dataset by correlating Landsat images with geospatial mineral deposit data, we will
attempt to implement and optimize a binary classifier which predicts whether a given Landsat image
shows land containing mineral deposits. If we successfully implement this binary classifier, we will
attempt to implement and optimize a multiclass model which predicts for a given Landsat image the set of
mineral deposits contained in the land.  

Landsat dataset: https://console.cloud.google.com/marketplace/product/usgs-public-data/landast

Supplementary mineral dataset: https://www.kaggle.com/datasets/thedevastator/mineral-ores-around-the-world

# How to use the TensorFlow container:

The Dockerfile in this repo defines a container you can use for GPU-accelerated training.
The container is the official TensorFlow container, extended to include the libraries our scripts require.
It may require an Nvidia GPU. Instructions to use on WSL2 with an Nvidia card as as follows:

[Make sure your WSL2 is configured for nvidia GPU support.](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) If you installed or updated your gpu driver recently it should already be visible within WSL. You can check using the command:
```
nvidia-smi
```

In the root of the repo, build the docker image: 
```
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t satimg-tf-image .
```

Use the image to start the container: 
```
docker run --gpus all -u $(id -u):$(id -g) -it -v $(pwd):/satimg-classifier satimg-tf-image
```

This will open a shell on the container. You will see the satimg-classifier directory in the home directory. This is the actual repo directory, mounted to the container. You should be able to read and write to the container, and run all of our scripts.

Verify that the container is accessing your GPU:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
