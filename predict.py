# predict.py
#
# PROGRAMMER: Brian Pederson
# DATE CREATED: 05/20/2019
# REVISED DATE: 05/28/2019
# PURPOSE: Produces an inference analysis of categories of flowers given an image and a checkpoint of a trained model.
#
# For parameters review docstring of get_input_args immediately below.
#
# Included functions:
#     get_input_args    - process input arguments specific to predict.py
#     predict           - core function which performs inference/prediction of image
#     predict_wrapper   - wrapper for predict which produces command line output
#     main              - main function performs prediction operation
#
##

# Imports python modules
import argparse
import os
import sys
import torch

# imports project shared utilities
import predict_utils
from predict_utils import build_model
from predict_utils import load_checkpoint
from predict_utils import process_image


# Argument parser utility specific to predict.py
def get_input_args():
    """
    Retrieves and parses the two mandatory command line arguments and four optional
    arguments provided by the user when they run the program from a terminal window.

    Mandatory command line arguments:
      1. Image Path as image_path - filename (and optional path) to image file
      2. Checkpoint as checkpoint_filepath - filename (and optional path) to checkpoint file
    Optional command line arguments:
      3. Top KKK most likely classes as --top_k with default value 5
      4. Category Names as category_names - filename (and optional path) to JSON file as --category_names
                          with default value 'cat_to_name.json'
      5. Flag to use GPU as --gpu with effective default False
                          Note: if no GPU is available this argument is ignored.
      6. Flag for "verbose" output as --verbose with effective default False

    This function returns these arguments as an ArgumentParser object.
    Parameters:
      None - using argparse module to create & store command line arguments
    Returns:
      parser.namespace - data structure that stores the command line arguments object
    """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.prog = 'predict.py'
    parser.description = "Predicts image category when given an image and trained model."

    # Argument 1
    parser.add_argument('image_path', type = str,
                        help = "filename (and optional path) to image file to be analyzed")
    # Argument 2
    parser.add_argument('checkpoint_filepath', type = str,
                        help = "filename (and optional path) to checkpoint file")
    # Argument 3:
    parser.add_argument('--top_k', type = int, default = '5',
                        help = "Number of KKK most likely classes to return")
    # Argument 4:
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = "File path to JSON file containing a mapping of categories to real names")
    # Argument 5:
    parser.add_argument('--gpu', action="store_true",
                        help = "Flag to enable GPU to perform inference operation if available")
    # Argument 6:
    parser.add_argument('--verbose', '-v', action="store_true",
                        help = "Flag to enable verbose (debugging) outpout")

    # Note: this will perform system exit if argument is malformed or imcomplete
    in_args = parser.parse_args()

    # return parsed argument collection
    return in_args


# performs core logic of inference/prediction on image using trained model.
def predict(image_path, model, top_k = 5, gpu = False, verbose = False):
    """ Predict the class (or classes) of an image using a trained deep learning model.
        Parameters:
          image path - filepath to image to be analyzed
          model - trained model to use for inference/prediction
          top KKK - specify the number of top K most likely class results to produce
          GPU flag - flag to enable use of GPU if available
          verbose flag - flag to enable verbose output

        Returns:
          probabilities - list of probabilities
          numeric classes - list of classes in the form of numeric codes
          name classes - list of classes in the form of name strings
    """

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    if verbose: print(f"Running in {device} mode.")

    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = torch.FloatTensor(image).to(device)

    #image.requires_grad_(False)
    with torch.no_grad():
        image.unsqueeze_(0)
        logps = model.forward(image)
        ps = torch.exp(logps)
        ps = ps.cpu()   # just force this to cpu always since remaining computations are trivial?
        top_probs, top_labels = ps.topk(top_k, dim=1)

        probs = top_probs.numpy()[0].tolist()
        labels = top_labels.numpy()[0].tolist()

        num_classes = [model.idx_to_class[x] for x in labels]
        name_classes = [model.cat_to_name[str(x)] for x in num_classes]

    return probs, num_classes, name_classes


# wrapper for predict which produces terminal/command line style output
def predict_wrapper(image_path, model, top_k = 5, gpu = False, verbose = False):
    """ Wrapper function for predict which produces terminal/command line style output

        Parameters:
          image path - filepath to image to be analyzed
          model - trained model to use for inference/prediction
          top KKK - specify the number of top K most likely class results to produce
          GPU flag - flag to enable use of GPU if available
          verbose flag - flag to enable verbose output

        Returns:
          None  - note this generates output to the console
    """

    probs, num_classes, name_classes = predict(image_path, model, top_k = top_k, gpu=gpu)

    # determine actual flower category based on path - by convention category is last subdirectory above image files.
    num_actual = image_path.split('/')[-2]
    name_actual = model.cat_to_name[num_actual]

    print(f"Actual image Class Number: {num_actual} - Class Name: {name_actual}")
    print(f"Top KKK {top_k} inference results:")

    for i in range(len(probs)):
        print(f"  Probability: {100*probs[i]:4.1f} - Class Number: {num_classes[i]:>3} - Class Name: {name_classes[i]}")

    if num_actual==num_classes[0]:
        print("Inference match.")
    else:
        print("Inference mis-match.")
    print()

    return None


# main function
def main():
    """ Main function contains core logic for predict.

        For parameters review docstring of get_input_args above.

    """

    in_args = get_input_args()
    if in_args.verbose: print(in_args)               # temp debug

    # throw errors right away for obvious errors - avoids deep traceback stack
    if not os.path.isfile(in_args.image_path):
        sys.exit(f"predict.py: error: argument image_path '{in_args.image_path}' file does not exist.")
    if not os.path.isfile(in_args.checkpoint_filepath):
        sys.exit(f"predict.py: error: argument checkpoint_filepath '{in_args.checkpoint_filepath}' file does not exist.")

    # Build model from checkpoint
    state_dict = load_checkpoint(in_args.checkpoint_filepath)
    model, criterion, optimizer = build_model(state_dict['arch'], hidden_sizes = state_dict['hidden_sizes'])

    if in_args.verbose: print(model.classifier)     # temp debug

    # restore state into various model components - perhaps should pass state into build_model()???
    model.load_state_dict(state_dict['state_dict'])
    model.idx_to_class = state_dict['idx_to_class']
    model.cat_to_name = state_dict['cat_to_name']

    optimizer.load_state_dict(state_dict['state_optimizer'])

    print()
    print(f"Utilizing architecture {state_dict['arch']} trained over {state_dict['tot_epochs']} epochs.")

    # perform the inference/prediction
    predict_wrapper(in_args.image_path, model, top_k = in_args.top_k, gpu = in_args.gpu, verbose = in_args.verbose)

    return None


# Call to main function to run the program
if __name__ == "__main__":
    main()

