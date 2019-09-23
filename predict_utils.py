# predict_utils.py
#
# PROGRAMMER: Brian Pederson
# DATE CREATED: 05/20/2019
# REVISED DATE: 05/28/2019
# PURPOSE: Module contains several common functions shared by predict.py and train.py
#
# Included functions:
#     get_idx_to_class  - utility to create cross map of index to class
#     get_cat_to_name   - utility to create cross map of numeric category to name category
#     build_model       - build a model based on densenet 121 or vgg16 pre-trained neural nets
#     save_checkpoint   - save checkpoint state
#     load_checkpoint   - load checkpoint state
#     process_image     - process image (i.e. tensor vs pil formats)
#     main              - main stub
#
##

# imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
import numpy as np
from PIL import Image

# define constants for two model architecture choices: densenet121; vgg16
ARCH_DENSE = 'densenet121'
ARCH_VGG = 'vgg16'


# Produce reverse crossmap of dataset.class_to_idx
def get_idx_to_class(image_dataset):
    """ Create cross map containing the index to class information.

        Parameters:
          image dataset  - dataset containing training images in class/image format
                           Note: can be any of the 'train', 'valid', or 'test' subdirectories.

        Returns:
          index to class (dict) - contains raw index provide by torch data loaders cross mapped to true numeric class values
    """

    # This is somewhat poorly documented via PyTorch.
    # i.e. class DatasetFolder() attribute class_to_idx: Dict with items (class_name, class_index).
    # PyTorch returns raw classification labels (class_index) based on an alphabetical sort of the class subdir names
    # then provides a reverse crossmap of the class subdirectory names (class_name) to the raw classification labels.

    # harvest crossmap from any one of the data subdirectories (e.g. train, valid, test)
    class_to_idx = image_dataset.class_to_idx

    # the reverse crossmap from raw classification label (class_index) to class subdirectory name (class_name)
    # is more useful since it is used to obtain class subdirectory names which represent the classes/labels
    idx_to_class = {}     # note this only works because the dictionary values are also unique in this case
    for key, val in class_to_idx.items():
        idx_to_class[val] = key

    return idx_to_class


# Load JSON file containing category number to category name mapping
def get_cat_to_name(category_names = 'cat_to_name.json'):
    """ Load cross map containing category to names information.

        Parameters:
          category names JSON file

        Returns:
          category to name (dict) - contains numeric categories as keys and category names (i.e. flower names) as values
    """

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    #for i in sorted(cat_to_name, key=int):
    #    print(f"Flower category: {i} - {cat_to_name[i]}")

    return cat_to_name


# Build the model reusing pre-trained features stage and swapping in custom classifier stage
def build_model(arch, learning_rate = 0.001, dropout = 0.1, hidden_sizes = None, verbose = False):
    """ Build model based on densenet 121 or vgg16 pre-trained neural nets using specified parameters.

        Parameters:
          architecture - one of two values specifying a pre-trained neural network to build upon (densenet121, vgg16).
          learning rate - learning rate for training algorithm. Defaults to 0.001 but can be varied as needed.
          dropout - dropout value for training algorithm. Defaults to 0.1 but can be varied as needed.
          hidden sizes - three element tuple containing number of elements of hidden layers.

        Returns:
          model - neural net model consisting of pre-trained stage and augmented project specific classifer stage
          criterion - criterion object  (hard-coded and set to nn.NLLLoss)
          optimizer - optimizer object  (hard-coded and set to optim.Adam)
    """

    # These are effectively constants for this school project. In a real project these would probably be parameterized.
    NUM_FLOWER_CAT = 102
    VGG_INPUT_SIZE = 25088
    DENSE_INPUT_SIZE = 1024

    # First import pretrained model.
    if arch==ARCH_VGG:
        model = models.vgg16(pretrained=True)
    elif arch==ARCH_DENSE:
        model = models.densenet121(pretrained=True)
    #if verbose: print(model)                # temp debug

    # Freeze parameters of features stage so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Hyperparameters for network
    output_size = NUM_FLOWER_CAT        # 102
    #dropout = 0.1                        # 0.1 - 0.5
    #learning_rate = 0.001                # 0.003

    if arch==ARCH_VGG:
        input_size = VGG_INPUT_SIZE
        if hidden_sizes==None:
            hidden_sizes = [4096, 1024, 256]    # use these as defaults for vgg16
    elif arch==ARCH_DENSE:
        input_size = DENSE_INPUT_SIZE
        if hidden_sizes==None:
            hidden_sizes = [512, 256, 128]      # use these as defaults for densenet121

    # Build the feed-forward network
    # Note: It seems based on "knowledge board" posts that this is  overbuilt; could/should have been one simple level...
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_sizes[2], output_size),
                                     nn.LogSoftmax(dim=1))
    #if verbose: print(model)                # temp debug

    # set criterion
    criterion = nn.NLLLoss()

    # set optimizer; only train the classifier parameters; feature parameters are effectively frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


# Save checkpoint
def save_checkpoint(arch, model, optimizer, checkpoint_filepath, tot_epochs = None):
    """ Save checkpoint state

        Parameters:
          architecture - one of two values specifying a pre-trained neural network to build upon (densenet121, vgg16)
          model - neural net model consisting of pre-trained stage and dynamic classifer stage
          optimizer - optimizer object
          checkpoint filepath - filename (and optional path) to checkpoint file
          total epochs - total epochs the model has been processed (including previous runs)

        Returns:
          None  - note this saves model state to a checkpoint file
    """

    #model = model.to('cpu')  # force model to cpu - is this necessary? doesn't seem to be necessary or useful?

    state_dict = {'arch': arch,
                    'idx_to_class': model.idx_to_class,
                    'cat_to_name': model.cat_to_name,
                    'state_dict': model.state_dict(),
                    'tot_epochs': tot_epochs,
                    'state_optimizer' : optimizer.state_dict()
                   }
    torch.save(state_dict, checkpoint_filepath)

    return None


def load_checkpoint(checkpoint_filepath, gpu = False):
    """ Load checkpoint state

        Parameters:
          checkpoint filepath - filename (and optional path) to checkpoint file
          GPU flag - flag to enable use of GPU if available

        Returns:
          state dictionary -
    """

    # there is a glitch in the torch.load function that requires "magic" parameters to load in GPU/cuda mode.
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    if device=='cuda':
        state_dict = torch.load(checkpoint_filepath)
    else:             # magic parameters below are taken from PyTorch discussion board
        state_dict = torch.load(checkpoint_filepath, map_location=lambda storage, loc: storage)

    # reconstruct hidden_sizes by examining state of the classifier component of the model
    # Note: make sure these nodes remain consistent with any modifications to classifer component.
    state_dict['hidden_sizes'] = [len(state_dict['state_dict']['classifier.0.weight']),
                                    len(state_dict['state_dict']['classifier.3.weight']),
                                    len(state_dict['state_dict']['classifier.6.weight'])]

    return state_dict


# Process a PIL image for use in a PyTorch model
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array

        Parameters:
          image - image in PIL image format

        Returns:
          image - image in Numpy array format
    """

    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256, 256))

    # Calculate the left and lower: (256-224)/2 = 16
    # Calculate the right and upper: 256 - 224 = 240
    box = (16, 16, 240, 240)
    img = img.crop(box)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = np.array(img)/255
    img = (img - mean)/std
    img = img.transpose(2, 0, 1)

    return img


# TODO: find out if this is necessary or in any way useful on a utility module
def main():
    """ Main stub"""
    return None

# Call to main function to run the program
if __name__ == "__main__":
    main()



