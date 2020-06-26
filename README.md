# Face Recognition 
A face recognation bases on Eigenface and Fisherface.
And a face detector base on HAAR classifier.

## requirments
python 3
numpy
PIL
matplotlib
cv2

## Dataset 
Yale face dataset.
All images are resized into (48, 48) to make it easier to caculate eigen values in my experiment.
It contains 165 images.

N is the number of images randomly selected from each subject (N < 11). For each
experiment, take 15N images from the database as training samples, and the rest as testing
samples.

K is the dimensions of Eigenspace or fisherface(Dimension reducted face data).


## Eigenface
| accuracy | K=1 |K=5| K=10 | K=20 | K=30 | K=50 | K=100 | K=200|
|   ----  | ----  | ----  |  ----  | ----  | ----  | ----  | ----  | ----  |
| N=3	|0.3|0.6583|0.8|0.8333|0.825|0.825|0.825|0.825|
| N=5	|0.1667|0.6768|0.7111|0.8333|0.8333|0.8333|0.8333|0.8333|
| N=7	|0.15|0.6167|0.7333|0.7667|0.7833|0.7833|0.7833|0.7833|
| N=9	|0.1667|0.7666|0.8|0.8333|0.8667|0.8667|0.8667|0.8667|

## Fisherface

| accuracy | K=1 |K=5| K=10 | K=20 | K=30 | K=50 | K=100 | K=200|
|   ----  | ----  | ----  |  ----  | ----  | ----  | ----  | ----  | ----  |
| N=3	|0.225 |0.275|0.4167|0.6|0.6166|0.6667|0.7416|0.7583 |
| N=5	|0.2556 |0.5|0.6111|0.6555|0.7222|0.8|0.8333|0.8889|
| N=7	|0.1667 |0.35|0.5167|0.6167|0.75|0.7333|0.7667|0.8|
| N=9	|0.2333|0.6|0.6333|0.7333|0.7667|0.8333|0.9|0.9|

# Detecting 
###  Algorithm 1: Haar Cascade classfier

Haar Cascade classfier is a kind of ensemble boosting learning. It typically relies on Adaboost classifiers . It’s trained on hundreds of sample images and the effect of object detection is really good at that time.


The experiments are based on opencv. All faces in all images are detected.

###  Algorithm 2: Sliding window + (Eigenface) fisherface classifier
Using the sliding window algorithm, set the appropriate sliding window, scan the input image, resize the image in each scanning window to the appropriate size (48 * 48), transform it to the feature space with PCA matrix, and calculate the Euclidean distance with the face feature vector. Then set the appropriate threshold value, when the Euclidean distance is less than the threshold value, then judge the current frame as face.

The slide windows are in 3 different size(48, 48), (96, 96), (300,300) .
After the rectangle scaning,  we can optimize the result with NMS (Non-Maximum Suppression) to reduce valid rectangles.
Not Working well for the dataset only contains 15 different people. It is not in a enough scale for training. Here are the results.


All results above are in result folders.

