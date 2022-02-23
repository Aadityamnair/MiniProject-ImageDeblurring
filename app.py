import numpy as np
import os
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.utils import save_image
import time



from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = DeblurCNN().to(device)
PATH='C:/Users/ME18820/Desktop/ImageDeblurring/outputs/model.pth'
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((896,896)),
    transforms.ToTensor(),
])

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 896, 896)
    save_image(img, name)

def predict():
    img_path='input.jpg'
    blur_image = cv2.imread(img_path)
    blur_image =transform(blur_image)
    blur_image = blur_image.to(device)
    save_decoded_image(blur_image[None,...].cpu().data, name="static/blur.jpg")
    outputs = model(blur_image[None,...])
    print('predicted')
    save_decoded_image(outputs.cpu().data, name="static/output.jpg")
    print('saved')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save('input.jpg')
                predict()
                time.sleep(2) #add delay
                return render_template('predict.html')
        except:
            pass
        else:
            return 'some error'


if __name__ == '__main__':
    app.run(port=5002, debug=True)
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()