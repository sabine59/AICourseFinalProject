import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict
from PIL import Image
import numpy as np
import glob, os
import json

def get_input_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
   
   
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()


    # Create 6 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: that's a path to a folder
    parser.add_argument('--dir', type = str, default = 'flower_data', 
                    help = 'path to the folder of flower images') 
    # Argument 2: the CNN-model
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'CNN-model either "resnet50", or "vgg16" (default)') 
    # Argument 3: number of hidden layers
    parser.add_argument('--numHL', type = int, default = 2, 
                    help = 'number of hidden layers, 1 or 2 allowed') 
    # Argument 4: GPU support
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'gpu support wished, input True or empty string') 
    # Argument 5: number of hidden layers
    parser.add_argument('--learn', type = float, default = 0.001, 
                    help = 'learning rate') 
    # Argument 6: number of epochs for training
    parser.add_argument('--epochs', type = int, default = 6, 
                    help = 'number of epochs for training') 

    return parser.parse_args()

in_arg = get_input_args()
  
print(in_arg) 
localrun = True

# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  



data_dir = in_arg.dir  
    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#network = 'vgg16' #either VGG-16 or ResNet-50
network = in_arg.arch
numberHL = in_arg.numHL    # number of hidden layers 1, 2, or 3
learning_rate = in_arg.learn
epochs = in_arg.epochs
gpu_use = in_arg.gpu


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(data_dir + '/train', transform=training_transforms)
image_datasets_test = datasets.ImageFolder(data_dir + '/test', transform=data_transforms)
image_datasets_validate = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=64)
validationloader = torch.utils.data.DataLoader(image_datasets_validate, batch_size=64)




# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[12]:


# TODO: Build and train your network
if (network == 'vgg16'):
    #model = models.vgg16(pretrained=True)
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Enable backpropagation only for the classifier layer
    for param in model.parameters():
        param.requires_grad = False
        
    #print(model)
    if (numberHL == 1):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif (numberHL == 2):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(4096, 1000)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif (numberHL == 3):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 80192)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(80192, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(4096, 1000)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
                               
    model.classifier = classifier
    
elif (network == 'resnet50'):          
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Enable backpropagation only for the classifier layer
    for param in model.parameters():
        param.requires_grad = False
                                   
    #print(model)
    if (numberHL == 1):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1000)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif (numberHL == 2):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1536)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(1536, 1000)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    

    elif (numberHL == 3):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1792)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1792, 1536)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(1536, 1000)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))   
    model.fc = classifier

# check for GPU support
device = 'cpu'
if (localrun & gpu_use):
    device = 'mps'
else:
    if (torch.cuda.is_available() & gpu_use):
        device = 'cuda'


model.to(device)
criterion = nn.NLLLoss()
if (network == 'vgg16'):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
elif (network == 'resnet50'):
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
#print(model)    
# start training sequence
print("Start training, Network = {}, hidden layers = {}, learning rate = {}, GPU on = {}".
             format(network, str(numberHL), learning_rate, device))
accuracy_mean = 0.
step = 0
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for ii, (images, labels) in enumerate(trainloader):
        # Move images and labels tensors to the GPU
        #print(images.shape)
        images, labels = images.to(device), labels.to(device)
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (ii%10 == 0):
            print("Batch Nr.: ", ii)

    else:
        # do the validation
        # turn off gradients while validating
        with torch.no_grad():
            accuracy_mean = 0.
            test_loss = 0
            model.eval() # turn off dropout
            for images, labels in validationloader:
                step += 1
                # Move input and label tensors to the GPU
                images, labels = images.to(device), labels.to(device)
                # Get the class probabilities
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                # Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
                #print(ps.shape)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                print(f'Accuracy: {accuracy.item()*100}%')
                accuracy_mean += accuracy.item() *100
            accuracy_mean = accuracy_mean / step
            step = 0
            model.train() # turn on dropout
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Mean Test Accuracy: {:.3f}".format(accuracy_mean))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


# TODO: Save the checkpoint 

if(device != 'cpu'):
    model.to('cpu')
if (os.path.exists("./" + network) == False):
    os.mkdir("./" + network)
model.class_to_idx = image_datasets_train.class_to_idx
torch.save(model.state_dict(), network + '/checkpoint.pth')
torch.save(model.class_to_idx, network + '/class_to_idx.pth')
torch.save(e, network + '/epochs.pth')
torch.save(numberHL, network + '/num_HL.pth')

