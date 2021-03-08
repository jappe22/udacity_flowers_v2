import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import time
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from collections import OrderedDict
import os.path
import PIL
from PIL import Image
import numpy as np
import json


################### Input #######################

parser = argparse.ArgumentParser()
  
parser.add_argument('-im', nargs='?', default = 'flowers/test/40/image_04563.jpg')
parser.add_argument('-save_dir', nargs="?", default = "savedir")
parser.add_argument('-checkpoint_name', nargs="?", default = "checkpoint1.pth")
parser.add_argument('-device', nargs="?", default = 'gpu')
parser.add_argument('-topk', nargs="?", default = 5, type = int)
parser.add_argument('-cat2name', nargs="?", default = 'cat_to_name.json')

args = parser.parse_args()
im = args.im; device = args.device

if device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using GPU for prediction.') if torch.cuda.is_available() else print('Using cpu for prediction')
else:
    device = torch.device("cpu")
    print('Using cpu for prediction')

############### Cat to name mapping #######################    
    
with open(args.cat2name, 'r') as f:
    cat_to_name = json.load(f)
    
############### Define functions preprocess, predict, loadcheckpoint and plt #######################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    model.eval()
    image = process_image(image_path)    
    image = torch.from_numpy(np.array([image])).float()
    image = Variable(image)
    if cuda:
        image = image.cuda()
    probabilities_e = model.forward(image)
    probabilities = torch.exp(probabilities_e).data
    top_p = torch.topk(probabilities, topk)[0].tolist()[0]
    top_class_index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    x = []
    top_class = []
    for i in range(len(model.class_to_idx.items())):
        x.append(list(model.class_to_idx.items())[i][0])   
    for i in range(topk):
        top_class.append(x[top_class_index[i]]) 
    
    return top_p, top_class    

def load_checkpoint(sdir, check):
    checkpoint = torch.load(sdir+'/'+check, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
       
    return model      
    
##################### load checkpoint #####################    

if os.path.isfile(args.save_dir+'/'+args.checkpoint_name):
    print ("Checkpoint file exists. Loading checkpoint.")
    model = load_checkpoint(args.save_dir, args.checkpoint_name)
  
    model.to(device)
           
else:
    print("No checkpoint found, can't predict.")
    exit()

    
#######################            #################################    
    
print(" ")
print("Flower file: ", im)

top_p, top_class = predict(im, model, args.topk)
print(top_p)
print(top_class)
print([cat_to_name[i] for i in top_class])



    
