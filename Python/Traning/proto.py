# import flask
# from flask import Flask, render_template


# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Welcome to the Flask API!"
# if __name__ == '__main__':
#     app.run(debug=True)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import PIL 

# read image using pil
img = PIL.Image.open(r'train\fracture\image3_1185_png.rf.75c7c6e6625f0515687ff4fb3bb961a2.jpg')
# Show Image
img.show()
