########################### import ###########################
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




########################### parser ###########################
parser = argparse.ArgumentParser()
  
parser.add_argument('-data_dir', nargs='?', default = 'flowers')
parser.add_argument('-save_dir', nargs="?", default = "savedir")
parser.add_argument('-pretrained_model', nargs="?", default = "vgg13")
parser.add_argument('-learning_rate', nargs="?", default = 0.0004, type = float)
parser.add_argument('-input_size', nargs="?", default = 25088, type = int)
parser.add_argument('-hidden_size', nargs="?", default = 1024, type = int)
parser.add_argument('-output_size', nargs="?", default = 102, type = int)
parser.add_argument('-epochs', nargs="?", default= 1, type = int)
parser.add_argument('-gpu', nargs="?", default = 1, type = int)
parser.add_argument('-checkpoint_name', nargs="?", default = 'checkpoint1.pth')
args = parser.parse_args()

########################### startmessage ###########################

print("  ")
print("     Start the Flower training Module     ")
print("  ")

########################### loading data ###########################
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

########################### transforms ###########################
data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

data_transforms_test_valid = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


########################### Load the datasets with ImageFolder ###########################
image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_test_valid)
image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test_valid)


###########################  dataloaders ###########################
trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=32)


########################### Label mapping ###########################
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

########################### define model ########################### 

model = getattr(models, args.pretrained_model)(pretrained=True)

if args.gpu == 1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
for param in model.parameters(): 
    param.requires_grad = False

model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(args.input_size, args.hidden_size)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_size, args.output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss() 
model.to(device)
                       
   
################ loading checkpoint function ####################

def load_checkpoint(sdir, check):
    checkpoint = torch.load(sdir+'/'+check, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    passed_epochs = checkpoint['passed_epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.classifier = checkpoint['classifier']
       
    return model, optimizer, passed_epochs    
    
########################### loading checkpoint if availabe ########################### 
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.66)    

print('Found checkpoint:',args.save_dir+'/'+args.checkpoint_name)
use_checkpoint = input("Use checkpoint? Press y ----> ")

if use_checkpoint == 'y': 
    if os.path.isfile(args.save_dir+'/'+args.checkpoint_name):
        print ("Checkpoint file exists. Loading checkpoint and continuing training.")
        model, optimizer, passed_epochs = load_checkpoint(args.save_dir, args.checkpoint_name)
    else:
        print("No checkpoint found, training from scratch")
        passed_epochs = 0
else:
    print("Training from scratch")
    passed_epochs = 0
########################### training ###########################

if args.epochs <= passed_epochs:
    print("You already trained", passed_epochs , "epochs. Try more epochs.")
    exit()

for e in range(passed_epochs,args.epochs):
    steps = 0
    print_every = 10
    model.to(device)
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        model.train()
        steps += 1
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, args.epochs))
            
            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            model.eval () #switching to evaluation mode so that dropout is turned off
            
            # Turn off gradients for validation to save memory and computations
            with torch.no_grad():
                
                model.to(device)
    
                valid_loss = 0
                accuracy = 0
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()
                     
                
            print(
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validloader)*100),
                  "Learning rate: {:.10f}..".format(optimizer.param_groups[0]['lr']))
           
            running_loss=0
        scheduler.step()            
   

print("Training ended.")            


########################### Save model to checkpoint file ###########################

cp =  args.save_dir+'/'+args.checkpoint_name


model.class_to_idx = image_datasets_train.class_to_idx

checkpoint = {'input_size' : args.input_size,
              'hidden_size' : args.hidden_size,
              'output_size' : args.output_size,
              'pretrained_model': args.pretrained_model,
              'learning_rate' : args.learning_rate,
              'classifier' : model.classifier,
              'passed_epochs': args.epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, cp)


print("Model and parameters saved.")