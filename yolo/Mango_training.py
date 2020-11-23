!git clone https://github.com/pjreddie/darknet.git

!cd darknet; head Makefile

!sed -i "s/GPU=0/GPU=1/g" darknet/Makefile
!sed -i "s/OPENCV=0/OPENCV=1/g" darknet/Makefile
!sed -i "s/CUDNN=0/CUDNN=1/g" darknet/Makefile

!head darknet/Makefile

!cd darknet; make

!cd darknet; ./darknet

import matplotlib.pyplot as plt
import cv2

def showimg(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

from google.colab import drive

drive.mount('/content/drive')

%cd ./darknet
!ln -s '/content/drive/My Drive/' /mydrive

!cp /mydrive/Yolo/train.cfg ./cfg/
!cp /mydrive/Yolo/obj.names ./data/
!cp /mydrive/Yolo/obj.data ./data/
!cp /mydrive/Yolo/generate_train.py ./
!cp /mydrive/Yolo/generate_valid.py ./

!wget http://pjreddie.com/media/files/darknet53.conv.74

!ln -s '/content/drive/My Drive/Labeled/' /Labeled

!python generate_train.py
!python generate_valid.py

!./darknet detector train data/obj.data cfg/train.cfg /mydrive/Yolo/backup_new/train.backup

!ls

%cd darknet


