import json
import numpy as np


class DataLoaderTrainPerftest:
    def __init__(self, file):
        self.file = file

    def _load_data(self, file):
        if file is None:
            file =self.file
        with open(file, 'r') as f:
            data = json.load(f)
        self.N_datapoints = len(data)
        return data   

    def _get_step_values(self, data, index=0):
        x_values = np.zeros(self.N_datapoints)
        for a, dict in enumerate(data):
            x_values[a] = dict["NN_training_step"]
        return x_values

    def get_plot_data(self, file=None, keys=None, seperability_index=None):
        #seperability index should be either integer or iterable of integers or None if all values found are used
        data = self._load_data(file)
        x_values = self._get_step_values(data)
        #find keys list
        if keys is None:
            keys = ["fid_mean_no_outlier", "fid_tol_no_outlier",
                    "mean_N_cnot", "tol_N_cnot"]
        if type(keys) is not list:
            raise TypeError("keys should be a list of the keys to be read out from the data")
        sep_index_list, seperabilities = self._get_seperabilities(data, seperability_index)
        y_values = [self._get_key_values(data, key, sep_index_list)
                    for key in keys]
        return x_values, y_values, keys, seperabilities

    def _get_key_values(self, data, key, seperability_index):
        #seperability_index has to be list of integer(s)
        y_values = np.zeros((len(seperability_index), self.N_datapoints))
        for training_step, data_dict in enumerate(data):
            for k, sep_index in enumerate(seperability_index):
                y_values[k, training_step] = data_dict[key][sep_index]
        return y_values

    def _get_seperabilities(self, data, seperability_index):
        #determine list of all requested seperability indices and seperabilites
        if seperability_index is None:
            seperabilities = data[0]["seperabilities"]
            #Ugly None handling due to legacy code where exact seperability was not listed when only fully entangled was asked for
            if seperabilities is None:
                seperabilities = ["All Entangled"]
            seperability_index = list(range(len(seperabilities)))  
        try:    
            #if multiple seperabilities are asked for
            seperabilities = data[0]["seperabilities"]
            if seperabilities is None:
                seperabilities = ["All Entangled"]
            seperabilities = [seperabilities[index] for index in seperability_index]
            seperability_index_list = seperability_index
        except TypeError:
            #if only one is requested
            seperabilities = data[0]["seperabilities"]
            if seperabilities is None:
                seperabilities = ["All Entangled"]
            seperabilities = seperabilities[seperability_index]
            seperabilities = [seperabilities]
            seperability_index_list = [seperability_index]
        return seperability_index_list, seperabilities



