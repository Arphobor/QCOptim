import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PlotGeneration import DataPlotter


class CSV_IBMRes_Loader:
    def __init__(self):
        """
        Initialize the CSVLoader class.
        """


    def load_csv_files(self, file_paths):
        """
        Load the CSV files into pandas dataframes and store them in the dataframes list.

        :return: List of pandas dataframes.
        """
        dataframes = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dataframes.append(df)
            else:
                print(f"File {file_path} does not exist.")
        return dataframes
    
    def read_files_in_folder(self, folder_path):
        files = []
        for entry in os.scandir(folder_path):
            if entry.name.endswith(".csv"):
                files.append(entry)
        locations = [os.path.join(folder_path, file.name) for file in files]
        return locations



class IBMSIMResultsAnalyzer:
    def __init__(self):
        pass

    def run(self, dataframe, max_N_CNOT=None):
        """
        Run the IBM results analyzer on the given dataframe.

        :param dataframe: Pandas dataframe.

        :return: Dictionary with the data.
        """
        self._load_data(dataframe)

        self.N_gate_keys = ["N_entangling", 
                            "N_entangling_sampler", 
                            "N_entangling_sampler_no_readout", 
                            "N_entangling_noisy_rho", 
                            "N_entangling_ideal_rho"]
        max_N_CNOT_list = [np.max(self.data[key]) for key in self.N_gate_keys]
        self.max_N_CNOT = int(max(max_N_CNOT_list))
        if max_N_CNOT is not None and max_N_CNOT > self.max_N_CNOT:
            self.max_N_CNOT = max_N_CNOT

        self.properties = self._get_properties(self.data)
        self.is_outlier, self.outlier_rows = self._get_outliers()
        self.data_no_outlier = self._make_copy_no_outliers(self.outlier_rows)
        self.properties_no_outlier = self._get_properties(self.data_no_outlier)
        self.transpiling_effects = self.get_transpiling_effects(keys=self.N_gate_keys[0:2],
                                                                data=self.data,
                                                                max_N_CNOT=self.max_N_CNOT)
        self.transpiling_probs = self.get_transpiling_probs(self.transpiling_effects)
        results = {"data": self.data,
                   "properties": self.properties,
                   "outliers": self.outlier_rows,
                   "data_no_outlier": self.data_no_outlier,
                   "properties_no_outlier": self.properties_no_outlier,
                   "transpiling_effects": self.transpiling_effects,
                   "transpiling_probs": self.transpiling_probs}
        return results

    def _load_data(self, dataframe):
        """
        Load the data from the dataframe into the data dictionary as class attribute.

        :param dataframe: Pandas dataframe.

        :return: Dictionary with the data.
        """
        self.data = {}
        for column in dataframe.columns:
            self.data[column] = dataframe[column].values
        self.N_rows = len(self.data[column])
        return self.data
    
    def _get_properties(self, data_dict=None):
        """
        Extract mean, std, min, and max values from the data dictionary.

        :param data_dict: Dictionary of data.

        :return: Dictionary with the data.
        """
        if data_dict is None:
            data_dict = self.data
        characteristic_values = {}
        for key, value in data_dict.items():
            if isinstance(value[0], float):
                characteristic_values[key] = {
                    "mean": value.mean(),
                    "std": value.std(),
                    "min": value.min(),
                    "max": value.max(),
                    "confid_sample_counting": self._get_confidence_by_counting_samples(value)
                }
        return characteristic_values
    
    @staticmethod
    def _get_confidence_by_counting_samples(sample_values):
        N_values = len(sample_values)
        N_confid = 0.95*N_values
        if N_confid-np.floor(N_confid)<10**(-5):
            N_confid = int(np.floor(N_confid))
        else:
            N_confid = int(np.ceil(N_confid))
        sorted_vals = np.sort(sample_values)
        interval = [sorted_vals[0], sorted_vals[N_confid]]
        for i in range(N_values-N_confid):
            test_interval = [sorted_vals[i], sorted_vals[i+N_confid]]
            if test_interval[1]-test_interval[0]<interval[1]-interval[0]:
                interval = test_interval
        return interval

    def _get_outliers(self, outlier_cutoff=3):
        is_outlier = {}
        outlier_rows = np.zeros(self.N_rows, dtype=bool)
        for key, value in self.properties.items():
            if key not in self.N_gate_keys:
                is_outlier[key] = np.abs(self.data[key]-value["mean"]) > outlier_cutoff*value["std"]
        for row in range(self.N_rows):
            if any([is_outlier[key][row] for key in is_outlier.keys()]):
                outlier_rows[row] = 1
        return is_outlier, outlier_rows

    def _make_copy_no_outliers(self, outlier_rows):
        data_no_outlier = {}
        for key, value in self.data.items():
            data_no_outlier[key] = value[~outlier_rows]
        return data_no_outlier

    @staticmethod
    def get_transpiling_effects(keys, data, max_N_CNOT):
        transpiling_matrix = np.zeros((max_N_CNOT+1, max_N_CNOT+1))
        N_rows = len(data[keys[0]])
        for k in range(N_rows):
            for key in keys[1:]:
                x_ind = int(data[keys[0]][k])
                y_ind = int(data[key][k])
                transpiling_matrix[y_ind, x_ind] += 1
        # np.set_printoptions(linewidth=100)
        # print(transpiling_matrix)
        return transpiling_matrix

    @staticmethod
    def get_transpiling_probs(transpiling_matrix):
        #x-axis should be the original CNOT gate number, y-axis should be the transpiled CNOT gate number
        transpiling_probs = np.zeros(transpiling_matrix.shape)
        sum_over_y = np.sum(transpiling_matrix, axis=0)
        for i in range(transpiling_matrix.shape[1]):
            if sum_over_y[i]>0:
                for j in range(transpiling_matrix.shape[0]):
                    transpiling_probs[j,i] = transpiling_matrix[j,i]/sum_over_y[i]
        return transpiling_probs

class IBMResultsPlotter(DataPlotter):
    def __init__(self, figsize=[13,5], *args, **kwargs):
        self.N_make_fid_by_CNOT_plot = 0
        super().__init__(figsize=figsize, *args, **kwargs)

    def plot_transpiling_effects(self, transpiling_prob_list, N_transpiled_list,
                                 labels=None, x_limits=None, y_limits=None):
        self.addfig("Transpiling Effects", ncols=len(transpiling_prob_list), addtitle=False, sharey=True)
        for k, (probs, N_transpiled) in enumerate(zip(transpiling_prob_list, N_transpiled_list)):
            plot = self._make_transpiling_plot(probs, N_transpiled, k, 
                                               x_limits, y_limits)
            self.current_ax[k].set_xticks([0, 3, 6, 9, 12], labels=["0", "3", "6", "9", "12"], fontsize=12)
            if labels:
                self.current_ax[k].set_title(labels[k])
        self.current_ax[1].set_xlabel("Original CNOT Gate Number", fontsize=14)
        self.current_ax[0].set_ylabel("Transpiled CNOT Gate Number", fontsize=14)  
        self.current_ax[0].set_yticks([0, 3, 6, 9, 12, 15, 18, 21, 24], labels=["0", "3", "6", "9", "12", "15", "18", "21", "24"], fontsize=12)
        for ax in self.current_ax:
            if x_limits is not None:
                ax.set_xlim(x_limits)
            if y_limits is not None:
                ax.set_ylim(y_limits) 
        plt.tight_layout(w_pad=-10) 
        cbar = self.current_fig.colorbar(plot, ax=self.current_ax)
        cbar.set_label("Probability of Transpilation Outcome", fontsize=12)
        return self.current_fig

    def _make_transpiling_plot(self, transpiling_probs, N_transpiled, index,
                               x_limits, y_limits):
        if x_limits is not None:
            min_x = x_limits[0]+1
            max_x = x_limits[1]
        else: 
            min_x = 0,
            max_x = transpiling_probs.shape[1]
        if y_limits is not None:
            min_y = y_limits[0]+1
            max_y = y_limits[1]
        else: 
            min_y = 0,
            max_y = transpiling_probs.shape[0]
        plot = self.current_ax[index].imshow(transpiling_probs)
        self.current_ax[index].invert_yaxis()
        # for i in range(min_y, max_y):
        #     for j in range(min_x, max_x):
        #         if N_transpiled[i, j]>0:
        #             text = self.current_ax[index].text(j, i, int(N_transpiled[i, j]),
        #                         ha="center", va="center", color="k", fontsize=6)                
        return plot

    def compare_agents_by_run(self, results, agent_class, method_key, 
                              rm_outliers=False):
        result_key = "data"
        if rm_outliers:
            result_key += "_no_outlier"
        x_by_run = [np.arange(len(result[result_key][method_key])) for result in results]
        y_by_run = [result[result_key][method_key] for result in results]
        labels = [result["filepath"] for result in results]
        self.addfig("Fidelity " + agent_class + " " + method_key)
        self.add_multiplot(y_by_run, xdata=x_by_run, plottype="scatter", 
                           labels=labels, s=5, marker="x")
        self.current_fig.legend(loc="lower right")
        self.current_ax.set_xlabel("Fidelity")
        self.current_ax.set_ylabel("Run Index")
        return

    def compare_simulators_by_run(self, result, keys):
        x_by_run = [np.arange(len(result["data"][key])) for key in keys]
        y_by_run = [result["data"][key] for key in keys]
        self.addfig("Fidelity " + result["filepath"])
        self.add_multiplot(y_by_run, xdata=x_by_run, plottype="scatter", 
                           labels=keys, s=5, marker="x")
        self.current_fig.legend(loc="lower right")
        self.current_ax.set_xlabel("Run Index")
        self.current_ax.set_ylabel("Fidelity")
        return

    def plot_difference_two_keys(self, result, key_A, key_B):
        diff = result["data"][key_A] - result["data"][key_B]
        self.addfig("Difference " + result["filepath"] + "\n" + key_A + "-" + key_B)
        self.addplot(diff, plottype="scatter", s=5, marker="x")
        self.current_ax.set_xlabel("Run Index") 
        self.current_ax.hlines(0, 0, len(diff), colors="k", linestyles="dashed", label="Zero")
        self.current_ax.hlines(np.mean(diff), 0, len(diff), colors="r", linestyles="dashed", label="Mean")
        self.current_ax.legend()
        return

    def make_cdf_plot_compare_agents(self, results, agent_class, cdf_key):
        x_cdf_Unres = [np.sort(result["data"][cdf_key]) for result in results]
        y_cdf_Unres = [np.linspace(0, 1, len(cdf)) for cdf in x_cdf_Unres]
        labels = [result["filepath"] for result in results]
        self.addfig("CDFs " + agent_class + " " + cdf_key)
        self.add_multiplot(y_cdf_Unres, xdata=x_cdf_Unres, plottype="scatter",
                           labels=labels, s=5, marker="x")
        self.current_fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.8))
        return
    
    def make_cdf_plot_compare_simulators(self, result, keys):
        x_cdf = [np.sort(result["data"][key]) for key in keys]
        y_cdf = [np.linspace(0, 1, len(cdf)) for cdf in x_cdf]
        self.addfig("CDFs " + result["filepath"])
        self.add_multiplot(y_cdf, xdata=x_cdf, plottype="scatter", labels=keys,
                            s=5, marker="x")
        self.current_fig.legend(loc="lower right")
        self.current_ax.set_xlabel("Fidelity")
        self.current_ax.set_ylabel("CDF")
        return
    
    def make_fid_by_CNOT_plot(self, results, agent_class, fid_key,
                              rm_outliers=False, make_legend=False,
                              errorbar_method="sample_counting", addtitle=None,
                              offset=0., *args, **kwargs):
        title = fid_key + " reached dependent on CNOT gates"
        if title not in self.plots:
            self.addfig(title, addtitle=addtitle)
        result_key = "properties"
        if rm_outliers:
            result_key += "_no_outlier"
        if errorbar_method == "std":    
            self._errbar_by_std(results, agent_class, fid_key, result_key, 
                                offset=offset, *args, **kwargs)
        elif errorbar_method == "sample_counting":
            self._errbar_by_sample_counting(results, agent_class, fid_key, result_key, 
                                            offset=offset, *args, **kwargs)
        else:
            raise ValueError("Unknown errorbar method.")
        #self.current_ax.set_xlabel("Average Number of CNOT gates (before transpiling)", fontsize=14)
        self.current_ax.set_xlabel("Number of CNOT gates", fontsize=14)
        self.current_ax.set_ylabel("Fidelity", fontsize=14)
        #self.current_ax.set_ylabel(fid_key, fontsize=14)
        self.current_ax.tick_params(axis='x', which='major', labelsize=12)
        self.current_ax.tick_params(axis='y', which='major', labelsize=12)
        if make_legend:
            self.current_fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.1), fontsize=14, ncols=1)
        self.N_make_fid_by_CNOT_plot += 1
        return

    def _errbar_by_std(self, results, agent_class, fid_key, result_key, offset=0.,
                       *args, **kwargs):
        x_CNOT = [result[result_key]["N_entangling"]["mean"] for result in results]
        x_CNOT = np.array(x_CNOT) + offset
        y_fid = [result[result_key][fid_key]["mean"] for result in results]
        
        y_errbar = [result[result_key][fid_key]["std"]*1.984 for result in results]
        self.current_ax.errorbar(x_CNOT, y_fid, yerr=y_errbar, fmt="+", label=agent_class,
                                 capsize=3.,*args, **kwargs)
        return
    
    def _errbar_by_sample_counting(self, results, agent_class, fid_key, result_key, offset=0,
                                   *args, **kwargs):
        x_CNOT = [result[result_key]["N_entangling"]["mean"] for result in results]
        x_CNOT = np.array(x_CNOT) + offset
        y_fid = [result[result_key][fid_key]["mean"] for result in results]

        y_err_intervals = [result[result_key][fid_key]["confid_sample_counting"] 
                           for result in results]
        y_errbar = [np.abs(interval-mean) for interval, mean in zip(y_err_intervals, y_fid)]
        y_errbar = np.array(y_errbar).transpose()
        self.current_ax.errorbar(x_CNOT, y_fid, yerr=y_errbar, fmt="+", label=agent_class,
                                 capsize=3., *args, **kwargs)
        return

def aggregate_results(filepath_list, max_N_CNOT):
    csv_loader = CSV_IBMRes_Loader()
    data_analyzer = IBMSIMResultsAnalyzer()
    dataframes = csv_loader.load_csv_files(filepath_list)
    results_list = [data_analyzer.run(df, max_N_CNOT=max_N_CNOT) 
                    for df in dataframes]
    for filepath, result in zip(filepath_list, results_list):
        result["filepath"] = filepath
    N_transpiled = [result["transpiling_effects"]
                    for result in results_list]
    N_transpiled = sum(N_transpiled)
    return results_list, N_transpiled


def main():
    #######5Q Quito#########
    files_Unres = []
    files_NNquito = ["Quito5Q/Quito5Q06CNOT/Perftests/fake_quito_results_69999_N100.csv",
                     "Quito5Q/Quito5Q07CNOT/Perftests/fake_quito_results_69999_N100.csv",
                     "Quito5Q/Quito5Q08CNOT/Perftests/fake_quito_results_69999_N100.csv",
                     "Quito5Q/Quito5Q09CNOT/Perftests/fake_quito_results_69999_N100.csv",
                     "Quito5Q/Quito5Q10CNOT/Perftests/fake_quito_results_67000_N100.csv",
                     "Quito5Q/Quito5Q11CNOT/Perftests/fake_quito_results_65000_N100.csv",
                     "Quito5Q/Quito5Q12CNOT/Perftests/fake_quito_results_100000_N100.csv"]
    files_Line = ["Line5Q/Line5Q06CNOT/Perftests/fake_manila_results_25000_N100.csv",
                  "Line5Q/Line5Q07CNOT/Perftests/fake_manila_results_30000_N100.csv",
                  "Line5Q/Line5Q08CNOT/Perftests/fake_manila_results_30000_N100.csv",
                  "Line5Q/Line5Q09CNOT/Perftests/fake_manila_results_31999_N100.csv",
                  "Line5Q/Line5Q10CNOT/Perftests/fake_manila_results_44998_N100.csv",
                  "Line5Q/Line5Q11CNOT/Perftests/fake_manila_results_91000_N100.csv",
                  "Line5Q/Line5Q12CNOT/Perftests/fake_manila_results_78000_N100.csv"]
    files_layered = ["BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_manilalayered_pairwiseU3_5Q_1L_100Nstate.csv",
                     "BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_manilalayered_pairwiseU3_5Q_2L_100Nstate.csv",
                     "BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_manilalayered_pairwiseU3_5Q_3L_100Nstate.csv"]
    files_quito = ["BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_quitolayered_pairwiseU3_5Q_1L_100Nstate.csv",
                   "BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_quitolayered_pairwiseU3_5Q_2L_100Nstate.csv",
                   "BenchmarkComparisons/LayeredIBMResults/Is5Q/fake_quitolayered_pairwiseU3_5Q_3L_100Nstate.csv"]

    min_N_CNOT = 0
    max_N_CNOT = 25
    data_analyzer = IBMSIMResultsAnalyzer()
    plotter = IBMResultsPlotter() 


    results_NNquito, N_transpiled_NNquito = aggregate_results(files_NNquito, max_N_CNOT)
    results_Line, N_transpiled_Line = aggregate_results(files_Line, max_N_CNOT)
    results_layered, N_transpiled_layered = aggregate_results(files_layered, max_N_CNOT)
    results_quito, N_transpiled_quito = aggregate_results(files_quito, max_N_CNOT)

    #plotter.compare_agents_by_run([results_NNquito[-1], results_layered[-1]], "NNquito", "fid_sampler",
    #                               rm_outliers=False)

    keys = ["fid_sampler", "fid_sampler_to_NN_approx", 
            "fid_sampler_no_readout", "fid_sampler_no_readout_to_NN_approx",
            "fid_noisy_state_sim", "fid_noisy_state_sim_to_NN_approx",
            "fid_ideal"]


    #####PAPER PLOT FOR FID IDEAL########
    fid_convergence_key = "fid_ideal"
    rm_outliers = True
    plt.rcParams['figure.figsize'] = [7, 4]
    plotter.make_fid_by_CNOT_plot(results_quito, "Layered Pairwise", fid_convergence_key,
                                  rm_outliers=rm_outliers, addtitle=False, color="red", offset=-0.1)
    plotter.make_fid_by_CNOT_plot(results_Line, "RL Manila", fid_convergence_key,
                                  rm_outliers=rm_outliers, addtitle=False, color="blue")
    plotter.make_fid_by_CNOT_plot(results_NNquito, "RL Quito", fid_convergence_key,
                                  rm_outliers=rm_outliers, make_legend=True, color="green", offset=+0.1)


    # plotter.make_fid_by_CNOT_plot(results_layered, "Layered Pairwise", fid_convergence_key,
    #                              rm_outliers=rm_outliers, make_legend=True, 
    #                              color="#2ca02c")
    plotfolder = "BenchmarkComparisons/Plots"
    plotter.current_ax.set_xlim([5.8, 12.3])
    plotter.current_ax.set_ylim([0.55, 1.01])
    plotter.save_currentfig(plotfolder, "FidIdealFull5Q")


    ######PAPER PLOT FOR FID SAMPLER NO READOUT########
    fid_convergence_key = "fid_sampler_no_readout"
    rm_outliers = False
    plt.rcParams['figure.figsize'] = [7, 4]
    plotter.make_fid_by_CNOT_plot(results_layered, "Layered Manila", fid_convergence_key,
                                  rm_outliers=rm_outliers, addtitle=False, color="k", offset=-0.1)
    plotter.make_fid_by_CNOT_plot(results_Line, "RL Manila", fid_convergence_key,
                                  rm_outliers=rm_outliers, addtitle=False, color="blue", offset=+0.03333)
    plotter.make_fid_by_CNOT_plot(results_quito, "Layered Quito", fid_convergence_key,
                                  rm_outliers=rm_outliers, addtitle=False, color="red", offset=-0.03333)
    plotter.make_fid_by_CNOT_plot(results_NNquito, "RL Quito", fid_convergence_key,
                                  rm_outliers=rm_outliers, make_legend=True, color="green", offset=+0.1)


    # plotter.make_fid_by_CNOT_plot(results_layered, "Layered Pairwise", fid_convergence_key,
    #                              rm_outliers=rm_outliers, make_legend=True, 
    #                              color="#2ca02c")
    plotfolder = "BenchmarkComparisons/Plots"
    plotter.current_ax.set_xlim([5.8, 12.3])
    plotter.current_ax.set_ylim([0.23, 0.9])
    plotter.save_currentfig(plotfolder, "FidSamplerNoReadoutFull5Q")

    plt.show()

    return


if __name__ == "__main__":
    main()
    
