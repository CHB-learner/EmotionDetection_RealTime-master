# EmotionDetection_Realtime
This is a Python 3 based project to display facial expressions (happy, sad, anger, fear, disgust, surprise, neutral) by performing fast & accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library.

The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dataset (if train)
Due to the limitations of upload size in github, I have uploaded the zip file of the dataset 'data.zip' on Baidu Netdisk.
Download the [data.zip]( https://pan.baidu.com/s/1uedy0S5e3WbdMLntLHDerQ?pwd=llt2 ) file and unzip it in the directory.

## Dependencies

1. Python 3.x, OpenCV 3 or 4, Tensorflow, TFlearn, Keras
2. Open terminal and enter the file path to the desired directory and install the following libraries
   * ``` pip install -r requirements```

   
## Execution

1. Unzip the 'data.zip' file in the same location（if train）
2. Open terminal and enter the file path to the desired directory and paste the command given below
3. For train: ``` python kerasmodel.py --mode display```     For display: ``` python kerasmodel.py --mode display```
