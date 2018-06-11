from os.path import join, isdir, isfile
from os import makedirs
from scipy.io import loadmat, savemat
from numpy import save, load
import pickle
import json


def save_variable(output_dir, output_filename, var, var_name=None,
                  file_type="p"):
    filename = join(output_dir, output_filename)

    if file_type == "mat":
        save_to_mat(filename, var_name, var)
    elif file_type == "npy":
        save_to_npy(filename, var)
    elif file_type == "p":
        save_to_pickle(filename, var)
    elif file_type == "json":
        save_to_json(filename, var)


def save_to_mat(output_filename, name, a):
    savemat(output_filename + ".mat", {name: a})


def save_to_npy(output_filename, a):
    if output_filename[-4:] != ".npy":
        output_filename += ".npy"
    save(output_filename, a)


def save_to_pickle(output_filename, o):
    if output_filename[-2:] != ".p":
        output_filename += ".p"
    pickle.dump(o, open(output_filename, "wb"))


def save_to_json(output_filename, var):
    if output_filename[-5:] != ".json":
        output_filename += ".json"
    with open(output_filename, 'w') as outfile:
        json.dump(var, outfile)


def load_variable(input_dir, input_filename, var_name=None, file_type="p"):
    filename = join(input_dir, input_filename)

    var = None

    if file_type == "mat":
        var = load_from_mat(filename, var_name)
    elif file_type == "npy":
        var = load_from_npy(filename)
    elif file_type == "p":
        var = load_from_pickle(filename)
    elif file_type == "json":
        var = load_from_json(filename)

    return var


def load_from_mat(input_filename, name):
    if input_filename[-4:] != ".mat":
        input_filename += ".mat"
    var_mat = loadmat(input_filename, squeeze_me=True)
    var = var_mat[name]
    return var


def load_from_npy(input_filename):
    if input_filename[-4:] != ".npy":
        input_filename += ".npy"
    var = load(input_filename)
    return var


def load_from_pickle(input_filename):
    if input_filename[-2:] != ".p":
        input_filename += ".p"
    var = pickle.load(open(input_filename, "rb"))
    return var


def load_from_json(input_filename):
    if input_filename[-5:] != ".json":
        input_filename += ".json"
    with open(input_filename) as infile:
        var = json.load(infile)
    return var


def my_makedir(path):
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise
