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
    Retrieves and parses the 3 command line arguments provided by the user when
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
    parser.add_argument('--imagepath', type = str, default = 'flower_data/valid/17/image_03829.jpg', 
                    help = 'path to the file with a flower images') 
    # Argument 2: the CNN-model
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'CNN-model either "resnet50", or "vgg16" (default)') 
 
    # Argument 3: GPU support
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'gpu support wished, input True or empty string') 

    # Argument 4: the filename of the json-file with the mapping directory to flowername
    parser.add_argument('--jsonpath', type = str, default = 'cat_to_name.json', 
                    help = 'path to the file with the mapping flower name to directory') 

    # Argument 5: number of top-probabilities to print
    parser.add_argument('--topk', type = int, default = 6, 
                    help = 'number of top-probabilities to print') 

    return parser.parse_args()



# TODO: Process a PIL image for use in a PyTorch model        
def process_image(image):
  ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
      returns an Numpy array
  '''

  size = 224, 224
  image.thumbnail(size)
  im_crop = image.crop((0, 0, 224, 224))
  im_crop.show()
  np_image = np.array(im_crop)
  np_image_new = np_image.astype('float32')/255
      
  means = np.array([0.485, 0.456, 0.406])
  stdev = np.array([0.229, 0.224, 0.225])
      

  np_image = (np_image_new - means) / stdev
      
  return np_image.transpose((2,0,1))

in_arg = get_input_args()
  
print(in_arg) 
localrun = True

network = in_arg.arch
gpu_use = in_arg.gpu
image_file = in_arg.imagepath
json_name_file = in_arg.jsonpath
numberTopKs = in_arg.topk




# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

with open(json_name_file, 'r') as f:
    cat_to_name = json.load(f, strict=False)
    
#print(cat_to_name)


# TODO: reconstructing the model
numberHL = 2
try:
    numHL = torch.load('./' + network + '/num_HL.pth')  
except:
    print("file not found", './' + network + '/num_HL.pth')   

if (network == 'vgg16'):
    #model = models.vgg16(pretrained=True)
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Enable backpropagation only for the classifier layer
    for param in model.parameters():
        param.requires_grad = False
        
    #print(model)
    if (numberHL == 1):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 102)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.5)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif (numberHL == 2):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif (numberHL == 3):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 12544)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.5, inplace=False)),
                          ('fc2', nn.Linear(12544, 4096)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.5, inplace=False)),
                          ('fc3', nn.Linear(4096, 102)),
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
                          ('fc1', nn.Linear(2048, 102)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.5)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif (numberHL == 2):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1536)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(1536, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    

    elif (numberHL == 3):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1536)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(1536, 1024)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(1024, 102)),
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

# TODO: Write a function that loads a checkpoint and rebuilds the model

criterion = nn.NLLLoss()
try:   
    model.load_state_dict(torch.load('./' + network + '/checkpoint.pth'))
except:
    print("file not found", './' + network + '/checkpoint.pth') 

try: 
    model.class_to_idx = torch.load('./' + network + '/class_to_idx.pth')
except:
    print("file not found", './' + network + '/class_to_idx.pth') 

  
model.to(device)

# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[309]:




def predict(infile, model, num_topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # TODO: Display an image along with the top 5 classes
    
    with torch.no_grad():
        model.eval()
   
        # open the image at the path in PIL-format
        with Image.open(infile) as im:
            # resize, crop and normalize, convert to numpy-array
            new_image = process_image(im)
    
        # convert the numpy-array to tensor
        image_tensor = torch.from_numpy(new_image.astype('float32'))
        inputs = image_tensor.view(1,3,224,224)

        loss = model.forward(inputs.to(device))
        ps = torch.exp(loss.cpu())
        ps = ps.topk(num_topk)
        print(ps)
        print()
        # the probabilities are related to output classes
        classes = []
        
        indx = ps[1].numpy()
        probs = ps[0].numpy()

     
        i = 0
        while i < num_topk:
            v = indx[0][i]
            i+=1
            classes.append(v)
         
        # the classes are related to the directory numbers
        # via model.class_to_idx
        # The directory numbers are related to the flower names
        # via cat_to_name.json
        labels = []  
        i = 0
        for key, value in model.class_to_idx.items():
            for cl in classes:
                
                if cl==value:
                    i+=1
                    labels.append(cat_to_name[key])
                    print("Class: {}, Label: {}, Probability: {}". format(cl, labels[i-1], probs[0][i-1]))
                    
# pick up the picture and compute the probability   
data_dir = 'flower_data'
while 1>0:
    input("Please enter a key to continue ...")
    flower_class = str(np.random.randint(0,102))
    path_to_img = data_dir + '/' + 'valid' + '/' + flower_class + '/' + '*.jpg'     
    for infile in glob.glob(path_to_img):
        predict(infile, model, numberTopKs)
        print("name of flower: ", cat_to_name[flower_class ])
        
    


