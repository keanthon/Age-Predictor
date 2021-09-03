# Age-Predictor
Age predictor that takes in an image and return the predicted age of the person in the image

Predictor is built using Convolutional Neural Network with an adjustable number of filters through the Keras library and trained using the UTKFace dataset

rename.sh is included to rename the source image files for ease of formatting when feeding in the data.

facecropmine.py is included to crop the desired face out of source images before it is ran through the predictor.

# Instructions
1. Run pip install -r requirements.txt before proceeding
2. To retrain model with new images, put images in UTKFace folder and run "python3 run64.py". Make sure to name your images as "{age}_description.jpg"
  For example, if there is an image of a 5 years old boy, it should be named "5_andy.jpg"
3. To predict age for new 

Please put your image in the folder "test2", run the rename.sh script and then run facecropmine.py before running run64.py for prediction
