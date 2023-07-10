# EmotionDetection_Realtime
This is a Python 3 based project to display facial expressions (happy, sad, anger, fear, disgust, surprise, neutral) by performing fast & accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library.

The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

1. Python 3.x, OpenCV 3 or 4, Tensorflow, TFlearn, Keras
2. Open terminal and enter the file path to the desired directory and install the following libraries
   * ``` pip install -r requirements```

## Model(pth)
Due to the limitations of upload size in github, I have uploaded the zip file of the dataset 'data.zip' on Baidu Netdisk.
Download the [Lin_vgg16Net.pth]( https://pan.baidu.com/s/1M0zFnDj2LoCH1p-EN0fYpg?pwd=i11c ) file and unzip it in the directory.


## Execution

1. Open terminal and enter the file path to the desired directory and paste the command given below.
2. Ready for  [Lin_vgg16Net.pth].
3. For display: ``` python main.py```.
