import argparse, json, os, shutil
import importlib.util
import sys

from os import path

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

    if args.mode not in ['train', 'eval']:
        print(f'\n {args.mode} is not a valid mode. Switching to eval')
        args_dict['mode'] = 'eval'

    with open(args.configpath, 'r') as f: # load the config file (-c)
        parameters = json.load(f)

    parameters.update(args_dict) # adds the fed arguments to parameters

    experiment_path = os.path.join(os.getcwd(), 'experiments', args.name, 
        args.mode)
    try:
        os.makedirs(experiment_path) # creates the relevant experiment path
    except FileExistsError:

        print(f'\n"{args.name}/{args.mode}" already exists. ')
        decision = str.lower(input("Clear and overwrite? [y/N] \t")) or "n"
        
        if decision != 'y':
            print('Aborting')
            return -1

        # Deletes the entire folder and builds a new one
        shutil.rmtree(experiment_path)
        os.makedirs(experiment_path)

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

    if path.isfile(modelpack_fullpath):
        spec = importlib.util.spec_from_file_location(modelpack_name, 
            modelpack_fullpath)
    else:
        print (f'Model Pack {modelpack_name} in config file does not exist.' +
            '\nDefaulting to basemodelpack')

        modelpack_defaultpath = os.path.join(os.getcwd(),
        params['modelpack_folder_path'], 'basemodelpack.py')

        spec = importlib.util.spec_from_file_location('basemodelpack', 
            modelpack_defaultpath) 

    return spec.loader.load_module()


if __name__ == '__main__':
    params = parser()
    if params == -1:
        sys.exit()

    modelpack = load_modelpack()

    if params['mode'] == 'train':
        #pos_folder = params['pos_train_path']
        #neg_folder = params['neg_train_path']
        datapath = params['train_path']
    elif params['mode'] == 'eval':
        #pos_folder = params['pos_test_path']
        #neg_folder = params['neg_test_path']
        datapath = params['test_path']

    buffer_path = params.get("buffer_path")
    channels = params.get("channels")

    main_ds, val_ds = modelpack.preprocess(datapath, channels)
