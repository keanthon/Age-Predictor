# Age-Predictor

Age-Predictor is a Convolutional Neural Network (CNN) model built with Keras and TensorFlow that takes an image as input and returns the predicted age of the person.

The predictor features an adjustable number of convolutional filters and is trained on the [UTKFace dataset](https://susanqq.github.io/UTKFace/) using the Adam optimizer.

## Included Scripts
- **`run64.py`**: The main script to train the model or generate predictions.
- **`facecropmine.py`**: A utility script to detect and crop faces out of source images before feeding them into the predictor.
- **`test2/rename.sh`**: A shell script to bulk rename source image files into the required formatting convention for training.

## Prerequisites
Install the required Python packages before running any scripts:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training / Retraining the Model
To train the model from scratch or fine-tune it with new images:
1. Place your training images inside the `UTKFace` folder.
2. **Formatting Details**: Ensure your images are named following the convention: `{age}_description.jpg`.
   *Example: An image of a 5-year-old boy should be named `5_andy.jpg`.*
3. Run the training script:
   ```bash
   python3 run64.py
   ```
   The script will automatically save the best performing model weights as `best_model.hdf5`.

### 2. Predicting Age for New Images
Make sure you have a trained model (`best_model.hdf5`) available in the root directory.

1. Add the new images you want to test into the `test2` folder.
2. Navigate into the `test2` directory and run the renaming script:
   ```bash
   cd test2
   ./rename.sh
   cd ..
   ```
3. Run the face cropper utility to extract the faces:
   ```bash
   python3 facecropmine.py
   ```
   The cropped face images will be saved in the `testdata2` directory.
4. Run the predictor in evaluation mode:
   ```bash
   python3 run64.py -p
   ```
   A new file named `prediction.txt` will be generated containing the predicted age for each sequenced image.
