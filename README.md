# Age-Predictor
Age predictor that takes in an image and return the predicted age of the person in the image

Predictor is built using Convolutional Neural Network with an adjustable number of filters and trained on the UTKFace dataset using Adam optimizer

rename.sh is included in folder "test2" to rename the source image files for ease of formatting when feeding in the data.

facecropmine.py is included to crop the desired face out of source images before it is ran through the predictor.

## Instruction
Run ```pip install -r requirements.txt``` before proceeding

### Retraining
1. To retrain model with new images, put images in "UTKFace" folder and run ```python3 run64.py``` Make sure to name your images as  "{age}_description.jpg"
   For example, if there is an image of a 5 years old boy, it should be named "5_andy.jpg"

### Prediction
1. To predict age for new images. Add your images to the folder named "test2"
2. In your current Age-Predictor directory, ```cd test2``` and run ```./rename.sh``` script
3. Change back to the parent directory by running ```cd ..```, and run ```python3 facecropmine.py```
4. The cropped faces should appear in testdata2
5. Run ```python3 run64.py -p``` to get a prediction of your images. A file named "prediction.txt" should now appear

