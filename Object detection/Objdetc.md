# Object-detection using VGG16 and Tensorflow

This project demonstrates an object detection model using the VGG16 architecture with TensorFlow and Keras. The model is trained to detect airplanes in images. The implementation includes data gathering, data preprocessing, model building, model training, evaluation, and prediction.

## Getting Started

### Prerequisites

To run the code, you need the following dependencies:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras (>=2.0)
- OpenCV (>=4.0)
- NumPy (>=1.16)
- scikit-learn (>=0.22)

You can install the required packages using pip:

    pip install tensorflow keras opencv-python numpy scikit-learn

## Dataset
The dataset used for training contains images of airplanes and their corresponding bounding box annotations. The images and annotations are provided in CSV format. You need to place the images in the images folder and the annotations CSV file in the project's root directory `airplanes.csv`. 

Link to the dataset: 
    
    https://drive.google.com/drive/folders/1jXNy-Fr1F9gDboLjM4N9wlmWT2CLxK2x?usp=sharing


## Training
To train the object detection model, run the Object_detection_using_VGG16.ipynb Jupyter notebook is provided in this repository. The notebook guides you through the entire process of data preprocessing, model building, and model training.

After training, the model will be saved as detect_Planes.h5.

## Evaluation
The model's performance can be evaluated using the Mean Squared Error (MSE) loss calculated during training. 

## Prediction
To test the model's object detection capability, you can use the Prediction checking section in the `Object_detection_using_VGG16.ipynb` notebook. Replace imagepath with the path to your desired test image and run the cell to see the predicted bounding box on the image.
