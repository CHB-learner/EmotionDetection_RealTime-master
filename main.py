import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms

Z

# haar xml文件的引入
face_xml = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
eye_xml = cv2.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')

# 情绪识别
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 网络
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use device:",device)
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class_indict = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "neutral",
    "5": "sad",
    "6": "surprise"
}

# 创建网络
model = vgg(model_name="vgg16", num_classes=7).to(device)
# 加载权重
weights_path = "./Lin_vgg16Net.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))




# -----------------------------------------------------------------------------------
# ----------------------------------------主函数--------------------------------------
# -----------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, img = cap.read()

    # 计算haar特征和对图像进行灰度转化gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸识别的检测
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    print('检测到人脸数量：', len(faces))  # 检测当前的人脸个数

    # 绘制人脸，为检测到的每个人脸进行画方框绘制
    # -------------------------------人 脸 检 测-------------------------------
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 人脸识别
        roi_face = gray[y:y + h, x:x + w]  # 灰色人脸数据
        roi_color = img[y:y + h, x:x + w]  # 彩色人脸数据

        # 1 gray
    # -------------------------------双 目 注 视-------------------------------
        eyes = eye_xml.detectMultiScale(roi_face)  # 眼睛识别，图片类型必须是灰度图
        if len(eyes) == 2: # 双眼注视
            print('-------检测到目光注视!---------')  # 打印检测出眼睛的个数
            for (e_x, e_y, e_w, e_h) in eyes:  # 绘制眼睛方框到彩色图片上
                cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 2)

    # -------------------------------情 绪 识 别-------------------------------
    Emotion_detect = True

    if Emotion_detect and len(faces) >= 1:
        image = roi_color.copy()
        img_Image = Image.fromarray(np.uint8(image))
        # [N, C, H, W]
        Img = data_transform(img_Image)
        # expand batch dimension
        Img = torch.unsqueeze(Img, dim=0)
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(Img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        dic = {class_indict[str(predict_cla)]: predict[predict_cla].numpy()}
        print(dic,'\n')
        # img
        Emotion_class = 'Emotion:{}'.format(class_indict[str(predict_cla)])

        prob = 'prob:{:.2f}%'.format(predict[predict_cla].numpy()*100)
        cv2.putText(img, Emotion_class, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, prob, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        pass


    # 可视化
    cv2.imshow('dst', img)
    cv2.waitKey(5)
    # 退出
    if cv2.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv2.destroyAllWindows()

