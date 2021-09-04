import numpy as np
import cv2
import os
import glob
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

curr = os.getcwd()
data = curr + '/test2'
jpgCount = len(glob.glob1(data, "*.jpg"))
counter = 0
for i in range(jpgCount):
    print(counter)
    
    img_path = data + '/' + str(i) + '.jpg'
    print(img_path)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        continue

    faces = detector.detect_faces(img)
    print(faces)
    if len(faces)>0:
        for result in faces:
            x, y, w, h = result['box']
            print(result['box'])
            os.chdir(curr+'/testdata2')
            d = max(w,h)

            # axis 0 is y axis 1 is x
            cropped = img[y+h-int(1.1*w):y+h+int(0.1*w), x-int(0.1*w):x+int(1.1*w)]
            #cropped = img[x-d:x+w, y-d:y+w]
            
            if cropped.size:
                #if cropped.shape[0] == cropped.shape[1]:
                filename = str(counter) + '.jpg'
                counter += 1
                cv2.imwrite(filename, cropped)

