# train.py
#
# PROGRAMMER: Brian Pederson
# DATE CREATED: 05/24/2019
# REVISED DATE: 05/30/2019
# PURPOSE: Train a neural network modeling categories of flowers.
#
# For parameters review docstring of get_input_args immediately below.
#
# Included functions:
#     get_input_args    - function to process input arguments specific to train.py
#     load_data         - utility function to perform the torch data loading
#     train_model       - core function which performs training of model
#     main              - main function performs training operation
#
##

# imports python modules
import argparse
import os
import sys
from time import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# imports project shared utilities
import predict_utils
from predict_utils import build_model
from predict_utils import load_checkpoint
from predict_utils import save_checkpoint
from predict_utils import get_idx_to_class
from predict_utils import get_cat_to_name


# Argument parser utility specific to train.py
def get_input_args():
    """
    Retrieves and parses the two mandatory command line arguments and eight optional
    arguments provided by the user when they run the program from a terminal window.

    Note: Rubric specs implied the checkpoint could be optional however this is
          modified to be mandatory in order to enable the reuse vs. new mode logic.

    Mandatory command line arguments:
      1. Data Path as data_dir - path to image data directory
      2. Checkpoint as checkpoint_filepath - filename (and optional path) to checkpoint file
    Optional command line arguments:
      3. Epochs as --epochs with default value 1
      4. Learning Rate as --learning_rate with default value 0.001
      5. Dropout as --dropout with default value 0.1
      6. Batch Size as --batch_size with default value 64
                          Note: The learning algorithm is sensitive to this. Never use small batch sizes.
      7. Flag to use GPU as --gpu with effective default False.
                          Flag set True runs in GPU (aka cuda) mode; False runs in CPU mode.
                          Note: if no GPU is available this argument is ignored.
                          Note: for all practical purposes this must run in GPU mode as CPU mode is too slow.
      8. Architecture as --arch used to choose one of densenet121 and vgg16.
                          Note: This argument is only used when creating model in "new" mode.
      9. Hidden Sizes as --hidden_sizes is a three element tuple with default (512, 256, 128)
                          Note: This argument is only used when creating model in "new" mode.
      10. Flag for "verbose" output as --verbose with effective default False


    This function returns these arguments as an ArgumentParser object.
    Parameters:
      None - using argparse module to create & store command line arguments
    Returns:
      parser.namespace - data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.prog = 'train.py'
    parser.description = "Trains a neural net model to recognize categories of flowers."

    # Argument 1
    parser.add_argument('data_dir', type = str,
                        help = "path to image data directory to be analyzed")
    # Argument 2
    parser.add_argument('checkpoint_filepath', type = str,
                        help = "filename (and optional path) to checkpoint file")
    # Argument 3:
    parser.add_argument('--epochs', type = int, default = 1,
                        help = "Number of epochs to iterate the training process")
    # Argument 4:
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = "Learning rate for training process")
    # Argument 5:
    parser.add_argument('--dropout', type = float, default = 0.1,
                        help = "Dropout value for training process")
    # Argument 6:
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = "Batch size for loading data")
    # Argument 7:
    parser.add_argument('--gpu', action="store_true",
                        help = "Flag to enable GPU to perform training operation if available")
    # Argument 8:
    parser.add_argument('--arch', type = str, choices = ['densenet121', 'vgg16'], # default = 'densenet121',
                        help = "CNN Model Architecture (e.g. densenet121, vgg16")
    # Argument 9:
    parser.add_argument('--hidden_sizes', nargs=3, type = int, # default=(512, 256, 128),
                        help = "Three values of hidden_sizes (e.g. 512 256 128)")
    # Argument 10:
    parser.add_argument('--verbose', '-v', action="store_true",
                        help = "Flag to enable verbose (debugging) output")

    # Note: this will perform system exit if argument is malformed or imcomplete
    in_args = parser.parse_args()

    # perform supplemental validations

    # throw errors right away for obvious errors - avoids deep traceback stack
    if not os.path.isdir(in_args.data_dir):
        sys.exit(f"train.py: error: argument data_dir '{in_args.data_dir}' directory does not exist.")

    # augment in_args with derived pseudo argument indicating reuse mode (i.e. new vs. reuse)
    in_args.reuse_mode = os.path.isfile(in_args.checkpoint_filepath) # if checkpoint exists then reuse it (i.e. process additional epochs)

    if in_args.reuse_mode and in_args.arch:
        print(f"train.py: warning: In reuse mode the --arch argument {in_args.arch} is ignored and existing architecture is used.")

    if in_args.reuse_mode and in_args.hidden_sizes:
        print(f"train.py: warning: In reuse mode the --hidden_sizes argument {in_args.hidden_sizes} is ignored and existing hidden sizes config is retained.")

    if not in_args.reuse_mode and not in_args.arch:
        sys.exit(f"train.py: error: In new mode the --arch argument is mandatory.")

    if not in_args.reuse_mode and not in_args.hidden_sizes:
        sys.exit(f"train.py: error: In new mode the --hidden_sizes argument is mandatory.")

    if in_args.gpu and not torch.cuda.is_available():
        print(f"train.py: warning: gpu (cuda) mode requested but gpu is not available - processing reverts to cpu mode.")

    # return parsed argument collection
    return in_args


# Initialize loading for data. Note these use generators so cost is not incurred until fetches occur.
def load_data(data_dir = 'flowers', batch_size = 64, verbose = False):
    """ Function to encapsulate data loading.

        Parameters:
          data dir - path to image data directory
          batch size - batch size for loading data
          verbose - flag to enable verbose (debugging) output

        Returns:
          dataloaders - dictionary containing train, valid, test torch provided dataloaders
          image datasets - dictionary containing train, valid, test image torchvision provided datasets
          data_transforms - dictionary containing train, valid, test data torchvision provided transforms
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
          'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
          'valid': transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
           'test': transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) }

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                       'test': datasets.ImageFolder(test_dir,  transform=data_transforms['test'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
                    'test': torch.utils.data.DataLoader(image_datasets['test'],  batch_size=batch_size)}

    return dataloaders, image_datasets, data_transforms


# core logic for training model
def train_model(model, epochs, device, dataloaders, criterion, optimizer, verbose = False):
    """ Function to encapsulate model training loop.

        Parameters:
          model - pre-trained neural network model provided by torchvision and augmented with classification stage
          epochs - number of epochs to iterate the training process
          device - token with possible values "cpu" or "cuda" (which indicates GPU mode)
          dataloaders - dictionary containing train, valid, test torch provided dataloaders
          verbose - flag to enable verbose (debugging) output

        Returns:
          None
    """

    print_every = 10
    print(f"Training model for {epochs} {'epoch' if epochs==1 else 'epochs'}.")

    model.to(device)    # move the model to CPU or GPU memory as necessary
    model.train()       # put the model into training mode

    steps = 0
    running_loss = 0

    #from workspace_utils import active_session
    #with active_session():
        # do long-running work here

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} - Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

    return None


# main function
def main():
    """ Main function contains core logic for train.

        For parameters review docstring of get_input_args above.

    """

    start_time = time()

    # process arguments
    in_args = get_input_args()
    if in_args.verbose: print(in_args)               # temp debug

    # initialize loaders, datasets, transforms
    dataloaders, image_datasets, data_transforms = load_data(data_dir = in_args.data_dir,
                                                             batch_size = in_args.batch_size,
                                                             verbose = in_args.verbose)

    # determine GPU vs CPU mode
    device = torch.device("cuda" if torch.cuda.is_available() and in_args.gpu else "cpu")
    print(f"Running in {device} mode.")

    # initialize model as necessary in reuse vs. new mode
    if in_args.reuse_mode:    # build model from existing checkpoint
        print(f"train.py: info: checkpoint {in_args.checkpoint_filepath} exists - train.py will run in reuse mode.")

        # obtain state from checkpoint
        state_dict = load_checkpoint(in_args.checkpoint_filepath, gpu = in_args.gpu)

        arch = state_dict['arch']
        tot_epochs = state_dict['tot_epochs']
        hidden_sizes = state_dict['hidden_sizes']

        print(f"Utilizing architecture {arch} existing model trained over {tot_epochs} epochs.")


    else:                     # build model in initialization mode
        print(f"train.py: info: checkpoint {in_args.checkpoint_filepath} does not exists - train.py will run in new mode.")

        arch = in_args.arch
        tot_epochs = 0
        hidden_sizes = in_args.hidden_sizes

        print(f"Utilizing architecture {arch} to create new model with hidden_sizes {hidden_sizes}.")

    # build the model
    model, criterion, optimizer = build_model(arch,
                                              learning_rate = in_args.learning_rate,
                                              dropout = in_args.dropout,
                                              hidden_sizes = hidden_sizes,
                                              verbose = in_args.verbose)
    if in_args.verbose: print(model.classifier)      # temp debug

    # post process the model by adding various components to it
    if in_args.reuse_mode:    # restore or initialize state as necessary in reuse vs. new mode
        # restore state into various model components
        model.load_state_dict(state_dict['state_dict'])
        model.idx_to_class = state_dict['idx_to_class']
        model.cat_to_name = state_dict['cat_to_name']

        # TODO: debug this - optimizer state can not be reloaded on a consistent basis.
        # This is causing trouble with possible optimizer.Adam bug in which restoring state from cpu mode to cuda mode throws exception.
        # Ran out of time to debug this - however not using this doesn't seem to prevent the model from training.
        #optimizer.load_state_dict(state_dict['state_optimizer'])

    else:
        # augment model with index to class (category) and category to name mappings
        model.idx_to_class = get_idx_to_class(image_datasets['train'])
        model.cat_to_name = get_cat_to_name()

    # train the model by looping for epochs times
    train_model(model, in_args.epochs, device, dataloaders, criterion, optimizer, verbose = in_args.verbose)

    tot_epochs += in_args.epochs
    print(f"Completed {in_args.epochs} additional epochs for a cumulative total of {tot_epochs} {'epoch' if tot_epochs==1 else 'epochs'}.")

    # save the state to new or reuse(ed) checkpoint
    save_checkpoint(arch, model, optimizer, in_args.checkpoint_filepath, tot_epochs = tot_epochs)

    tot_time = time() - start_time # calculate difference between end time and start time
    print("** Total Elapsed Runtime:",
          f"{str(int((tot_time/3600)))}:{str(int((tot_time%3600)/60))}:{str(round((tot_time%3600)%60))}")

    return None

# Call to main function to run the program
if __name__ == "__main__":
    main()

