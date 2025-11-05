import os 
import numpy as np
import matplotlib.pyplot as plt
from PlotGeneration import DataPlotter
from Perftest import DataLoaderTrainPerftest
from scipy.optimize import curve_fit, minimize_scalar

class TrackedVarLoader:
    def __init__(self, folder="CheckpointTraining", variables=None):
        #variables should be a list of strings containing the descriptors of
        #variables that ought to be in the plot
        self.folder = folder
        self.file_type_str = ".npy"
        self.plotter = DataPlotter()
        self.run(variables=variables)
        return
    
    def run(self, variables):
        self._get_file_list()
        if variables is None:
            self._get_descriptors()
        else:
            self.tracked_variables = variables
        self._load_data()
        self._gen_plots()
        return self.data

    def _get_file_list(self):
        self.file_list = []
        with os.scandir(self.folder) as it:
            for entry in it:
                if entry.name.endswith(self.file_type_str) and entry.is_file():
                    self.file_list.append(entry.name)
        return self.file_list

    def _get_descriptors(self):
        self.tracked_variables = []
        file_type_len = len(self.file_type_str)
        for file in self.file_list:
            descriptor = self._extract_descriptor(file)
            #exclude files that dont have digits at all and dont have digits in
            #the last position of the file name
            if (descriptor != file 
                and file[-file_type_len-1].isdigit()
                and descriptor not in self.tracked_variables):
                self.tracked_variables.append(descriptor)
        return self.tracked_variables

    def _extract_descriptor(self, file_name):
        descriptor = ""
        for char in file_name:
            if char.isdigit():
                break
            descriptor += char
        return descriptor.strip()

    def _load_data(self):
        self.data = {}
        for descriptor in self.tracked_variables:
            N_files = self._get_N_of_files(descriptor)
            self.data[descriptor] = []
            for i in range(N_files):
                file_str = (descriptor + str(i) + self.file_type_str)
                location = os.path.join(self.folder, file_str)
                self.data[descriptor].append(np.load(location))
        return self.data

    def _get_N_of_files(self, descriptor):
        N_files = 0
        for file in self.file_list:
            if descriptor in file:
                N_files+=1
        return N_files

    def _gen_plots(self):
        for key, ydata_list in self.data.items():
            x_start = 0
            self.plotter.addfig(key)
            for ydata in ydata_list:
                N_points = ydata.shape[0]
                #if N_points >= 1:
                x_stop = x_start+N_points
                xdata = np.arange(start=x_start, stop=x_stop)
                self.plotter.addplot(ydata, xdata=xdata, plottype="scatter", 
                                     s=1)
                x_start = x_stop
        return
    
    def save_plots(self, folder=None):
        if folder is None:
            folder = self.folder
        self.plotter.save_plots(folder)
        return

class CDFConfidenceMinimizer:
    def __init__(self):
        return
    
    def run(self, samples, cut_outliers=True):
        self.samples = samples
        cleaned_samples = self._clean_input(cut_outliers)
        self.x_cdf, self.y_cdf = self._get_cumulative_distrib(cleaned_samples)
        self.get_fit_params(self.x_cdf, self.y_cdf)
        self.conf_interval = self._get_confid_interval()
        self.mean = np.mean(cleaned_samples)
        return self.mean, self.conf_interval
        
    def _clean_input(self, cut_outliers, N_std_tolerance=3):
        if cut_outliers:
            mean = np.mean(self.samples)
            tol = N_std_tolerance*np.std(self.samples)
            self.cutoff = [mean-tol, mean+tol]
            mask = np.abs(self.samples-mean)<=tol
            cleaned_samples = self.samples[mask]
        else:
            self.cutoff = None
            cleaned_samples = self.samples
        return cleaned_samples

    def _get_cumulative_distrib(self, samples):
        x_cdf = np.sort(samples)
        y_cdf = np.arange(len(samples))/len(samples)
        return x_cdf, y_cdf

    def _function_to_fit(self, x, A, K, Q, B, nu, M):
        #is generalize_logistic_function
        #see https://en.wikipedia.org/wiki/Generalised_logistic_function for function definition (C=1)
        denom = np.power(1 + Q*np.exp(-B*(x-M)), 1/nu)
        y = A + (K-A)/denom
        return y
    
    def _get_fit(self, xdata, ydata):
        x_mean = np.mean(xdata)
        return curve_fit(self._function_to_fit, xdata, ydata, 
                         p0=[0.,1.,1.,1.,1.,x_mean])
    
    def get_fit_params(self, xdata, ydata):
        self.fit_params = self._get_fit(xdata, ydata)[0]
        self.A = self.fit_params[0]
        self.K = self.fit_params[1]
        self.Q = self.fit_params[2]
        self.B = self.fit_params[3]
        self.nu = self.fit_params[4]
        self.M = self.fit_params[5]
        return self.fit_params
    
    def _fit_func_inverse(self, y):
        inner = np.power(((self.K-self.A)/(y-self.A)), self.nu)
        log_inner = (inner-1)/self.Q
        x = self.M-np.log(log_inner)/self.B
        return x
    
    def _get_confid_interval(self):
        conf_optim = self._optimize_confid_interval()
        yupper = conf_optim.x
        confid_upper = self._fit_func_inverse(yupper)
        confid_lower = self._fit_func_inverse(yupper-0.95)
        return [confid_lower, confid_upper]
    
    def _optimize_confid_interval(self):
        return minimize_scalar(self._confid_interval_optim_func, 
                                 bounds=[0.950001,1], method='bounded')
    
    def _confid_interval_optim_func(self, yupper):
        confid_upper = self._fit_func_inverse(yupper)
        confid_lower = self._fit_func_inverse(yupper-0.95) 
        return confid_upper-confid_lower  

    def _get_fit_plot(self, y_bounds=None):
        if y_bounds is None:
            y_bounds = [0,1]
        yfit = np.arange(start=y_bounds[0], stop=y_bounds[1], step=0.005)
        xfit = self._fit_func_inverse(yfit)
        return xfit, yfit
    
    def show_plots(self, hbars=None):
        plotter  = DataPlotter()
        plotter.addfig("Data (mean, confid interval, cutoff)")
        plotter.addplot(self.samples, plottype="scatter", s=1)
        x_max = len(self.samples)
        x_min = 0
        plotter.current_ax.hlines(self.mean, x_min, x_max, 
                                  linestyles="dashed", colors="blue", label="Mean")
        plotter.current_ax.hlines(self.conf_interval, x_min, x_max, 
                                  linestyles="dashed", colors="green", label="fit bounds")
        if self.cutoff is not None:
            plotter.current_ax.hlines(self.cutoff, x_min, x_max, 
                                      linestyles="dashed", colors="red", label="cutoff")   
        if hbars is not None:
            plotter.current_ax.hlines(hbars, x_min, x_max, 
                                      linestyles="dashed", colors="purple", label="bounds by counting")   
        plotter.current_ax.legend()
             
        yfit = np.arange(1, step=0.005)
        xfit = self._fit_func_inverse(yfit)
        plotter.addfig("Cumulative Distribution")
        plotter.addplot(self.y_cdf, xdata=self.x_cdf, plottype="scatter", s=1)
        plotter.addplot(yfit, xdata=xfit, plottype="plot", linestyle="--",
                        color="red")
        return plotter


def get_confidence_by_counting(samples, percent=0.95):
    sorted_samples = np.sort(samples)
    N_samples = len(sorted_samples)
    N_in_confidence = int(np.ceil(float(N_samples)*percent))
    N_indexes_to_test = N_samples-N_in_confidence+1
    min_interval = [sorted_samples[0], sorted_samples[-1]]
    for i in range(N_indexes_to_test):
        interval = [sorted_samples[i], sorted_samples[-N_indexes_to_test+i]]
        if (interval[1]-interval[0])<(min_interval[1]-min_interval[0]):
            min_interval = interval
    return min_interval

def plot_fids_and_confid(perftest_fids, interval_by_counting):
    print("Mean of the distribution is", np.mean(perftest_fids))
    print("Confidence interval by counting through the samples is ", interval_by_counting)
    plotter  = DataPlotter()
    plotter.addfig("Data (mean, confid interval, cutoff)")
    plotter.addplot(perftest_fids, plottype="scatter", s=1)    
    plotter.current_ax.hlines(interval_by_counting, 0, len(perftest_fids), 
                                      linestyles="dashed", colors="green", label="bounds by counting")   
    plotter.current_ax.hlines(np.mean(perftest_fids), 0, len(perftest_fids), 
                                      linestyles="dashed", colors="red", label="Mean") 
    outlier_bound = 5*np.std(perftest_fids)  
    no_outliers = perftest_fids[perftest_fids>(np.mean(perftest_fids)-outlier_bound)]
    print("Mean of the distribution without outliers is", np.mean(no_outliers), 
          "Outlier Cutoff is 5 times standard deviation")
    plotter.current_ax.hlines(np.mean(no_outliers), 0, len(perftest_fids), 
                                      linestyles="dotted", colors="purple", label="Mean no outliers") 
    plotter.current_ax.hlines(np.mean(no_outliers)-outlier_bound, 0, len(perftest_fids), 
                                      linestyles="dotted", colors="orange", label="Outliers cutoff") 
    plotter.current_ax.legend()
    plt.show()
    return


def main(folder, fids_location):
    import traceback
    import logging
    
    tracked_var_handler = TrackedVarLoader(folder=folder)
    tracked_var_handler.save_plots()

    ###################GETTING TRAIN PERFTEST PLOTS##################
    #try:
    train_perftest_file = os.path.join(folder, "Training_Perftest_Summary.json")
    train_perftests = DataLoaderTrainPerftest(train_perftest_file)
    x_values, y_values, keys, seperabilities = train_perftests.get_plot_data(seperability_index=None)
    plot_labels = [str(seperability) for seperability in seperabilities]
    plotter_perftest_results = DataPlotter()
    plotter_perftest_results.addfig("Train Perftests", nrows=len(keys))
    plotter_perftest_results.multi_axis_plot(keys, y_values, xdata=x_values)
    plotter_perftest_results.multi_axis_plots_add_labels(plot_labels)
    plotter_perftest_results.save_plots(folder)
    # except FileNotFoundError:
    #     print("No Training Perftest found at specified location")
    # except Exception as e:
    #     logging.error(traceback.format_exc())


    ##################GETTING MEAN AND STANDARD DEVIATION OF DISTRIBUTION##################
    # GNecessary to select correct filename based on desired step number and relativ path to folder!
    try:
        perftest_fids = np.load(fids_location)
    except FileNotFoundError:
        print("No Perftest Fids found at specified location")
    except Exception as e:
        logging.error(traceback.format_exc())    
    else:
        interval_by_counting = get_confidence_by_counting(perftest_fids)
        try:
            cdf_fitter = CDFConfidenceMinimizer()
            mean, confid_interval = cdf_fitter.run(perftest_fids)
            print("Mean of the distribution is", mean)
            print("Confidence interval of the distribution is given by", confid_interval)
            print("Plus minus distance to error bars is", confid_interval-mean)
            print("Confidence interval by counting through the samples is ", interval_by_counting)
            cdf_fitter.show_plots(hbars=interval_by_counting)
        except RuntimeError:
            plot_fids_and_confid(perftest_fids, interval_by_counting)  
        except Exception as e:
            logging.error(traceback.format_exc())      

    plt.show()
    return




if __name__ == "__main__":
    folder = "Training4Q/FNN_4Q_Unrestricted_6_CNOT"
    filename = "Perftests/PerftestFids.npy"
    fids_location = os.path.join(folder, filename)
    main(folder, fids_location)