import os 
import sys
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras import Input, Model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization, MaxPooling2D, Conv2D, Activation, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend

def main():
    if len(sys.argv) == 2:
        if sys.argv[1]=='-s':
            print("Predicting from 0 to 146")
            predict_score_img = []
            for i in range(39):
                path = 'testdata2/' + str(i) + '.jpg'
                # print(path)
                img = cv2.imread(path)
                img = cv2.resize(img, (128,128))
                predict_score_img.append(img)
            
            model = create_cnn(128, 128, 3, regress=True)
            model.load_weights("best_model.hdf5")

            opt = Adam(lr=5e-3, decay=1e-3 / 200)
            model.compile(loss='mean_absolute_percentage_error', optimizer=opt)
            print("Loaded previously trained best weights")
            predict_score_img = np.array(predict_score_img) / 255.0
            predict_rating = 110 * model.predict(predict_score_img).flatten()
            
            for i, rating in enumerate(predict_rating):
                print(f"{i} :    {rating}")
        
        else:
            print("My life")
            predict_score_img = []
            for i in range(5):
                path = 'mylife/' + str(i) + '.jpg'
                img = cv2.imread(path)
                img = cv2.resize(img, (128,128))
                predict_score_img.append(img)
            
            model = create_cnn(128, 128, 3, regress=True)
            model.load_weights("best_model.hdf5")

            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model.compile(loss='mean_absolute_percentage_error', optimizer=opt)
            print("Loaded previously trained best weights")
            predict_score_img = np.array(predict_score_img) / 255.0
            predict_rating = 110 * model.predict(predict_score_img).flatten()
            
            for i, rating in enumerate(predict_rating):
                print(f"{i} :    {rating}")
        



    else:
        
        print("Loading image...")
        images = []
        ages = []
        dirt = 'crop_part1/'
        for image_path in os.listdir(dirt):
            img = cv2.imread(dirt+image_path)
            print(image_path)
            try:
                img = cv2.resize(img, (128,128))
                images.append(img)
                age = float(image_path.split('_')[0])
                print(age)
                ages.append(age)
        
            except Exception:
                continue
               
        
        # normalize
        images = np.array(images) / 255.0
        ages = np.array(ages).T 
        print(ages)
        ages/=110.0
        
        #print(images)
        # split data
        X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.15, random_state=42)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        model = create_cnn(128, 128, 3, regress=True)

        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

        print("Training...")
        # save checkpt
        checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
        save_best_only=True, mode='auto', period=1)
        callbacks_list = [checkpoint]
        model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8, callbacks=callbacks_list)

        print("[INFO] predicting age...")
        preds = model.predict(X_test)
        diff = preds.flatten() - y_test

        print(diff)


def create_cnn(width, height, depth, filters=(16, 32, 128), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        # filter is number of filter, 3x3 window size
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

if __name__ == '__main__':
    main()