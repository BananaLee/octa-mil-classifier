import argparse
import json
import os
import importlib.util

""" DOCSTRING GOES HERE"""

def parser():
    """
    Creates a new folder with relevant experiments based on the arguments
    given and the loaded config file.

    In the folder, also produces a config file to archive the config used.

    Returns a dictionary of parameters
    """

    parser = argparse.ArgumentParser(description='OCT-A Classifier Zeisslab')

    parser.add_argument("-n", "--name", type=str, default='test',
        help="Experiment name")

    parser.add_argument("-c", "--configpath", default='./config.json', 
    	help="Path to JSON file providing arguments")

    parser.add_argument("-m", "--mode", default='eval', 
        help="Mode of running (train or eval)")

    args = parser.parse_args() # access via args.argument
    args_dict = vars(args) # access via args_dict["argument"] 

    with open(args.configpath, 'r') as f: # load the config file (-c)
        parameters = json.load(f)

    parameters.update(args_dict) # adds the fed arguments to parameters

    experiment_path = os.path.join(os.getcwd(), 'experiments', args.name, 
        args.mode)
    try:
        os.makedirs(experiment_path) # creates the relevant experiment path
    except FileExistsError:
        print('\n"{}" already exists.'.format(args.name), end='\t')
        decision = str.lower(input("Overwrite? [y/N] \t")) or "n"
        if decision != 'y':
            print('Aborting')
            return -1

    with open(os.path.join(experiment_path, 'config.json'), 'w+') as f:
        json.dump(parameters, f, sort_keys=True, indent=0) # writes config

    return parameters

    
def load_modelpack():
    """
    Loads a self-contained model pack which should contain a preprocess
    function, train function, prediction function, and model architecture. 

    Defaults to basemodelpack which is provided as a basic CNN module
    """

    try:
        modelpack_name = params['modelpack_used']
    except:
        modelpack_name = 'basemodelpack'

    modelpack_fullpath = os.path.join(os.getcwd(),
        params['modelpack_folder_path'], modelpack_name+'.py')

    spec = importlib.util.spec_from_file_location(modelpack_name, 
        modelpack_fullpath)

    return spec.loader.load_module()


if __name__ == '__main__':
    params = parser()
    modelpack = load_modelpack()

    #modelpack.preprocess()
