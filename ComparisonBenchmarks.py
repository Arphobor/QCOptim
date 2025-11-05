import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import time

from IBMResultsplots import IBMSIMResultsAnalyzer, IBMResultsPlotter, CSV_IBMRes_Loader

def plot_results_transpiling(backend):
    data_folder = "BenchmarkComparisons/TranspilingEffectsOnly"
    keys = ["N_entangling", "N_entangling_sampler"]
    max_N_CNOT = 40
    qubitnumber = 5
    csv_loader = CSV_IBMRes_Loader()
    analyzer = IBMSIMResultsAnalyzer()
    plotter = IBMResultsPlotter(figsize=[10,5])
    files = csv_loader.read_files_in_folder(data_folder)
    files = [file for file in files if (str(qubitnumber)+"Q") in file]

    Unrestricted_files = [file for file in files if backend+"Unrestricted" in file]
    Unrestricted_results = csv_loader.load_csv_files(Unrestricted_files)
    transpiling_matrices = [analyzer.get_transpiling_effects(keys, result, max_N_CNOT)
                            for result in Unrestricted_results]
    N_transpiled_Unres = sum(transpiling_matrices)

    Line_files = [file for file in files if backend+"Line" in file]
    Line_results = csv_loader.load_csv_files(Line_files)
    transpiling_matrices = [analyzer.get_transpiling_effects(keys, result, max_N_CNOT)
                            for result in Line_results]
    N_transpiled_Line = sum(transpiling_matrices)

    T_files = [file for file in files if backend+"T" in file]
    T_results = csv_loader.load_csv_files(T_files)
    transpiling_matrices = [analyzer.get_transpiling_effects(keys, result, max_N_CNOT)
                            for result in T_results]
    N_transpiled_T = sum(transpiling_matrices)

    N_transpiled_list = [N_transpiled_Unres, N_transpiled_Line, N_transpiled_T]
    transpiling_matrix_list = [analyzer.get_transpiling_probs(N_transp) 
                               for N_transp in N_transpiled_list]
    plotter.plot_transpiling_effects(transpiling_matrix_list, N_transpiled_list,
                                     labels=["Unrestricted Circuits\ntranspiled to Quito", "Manila Circuits\ntranspiled to Quito", "Quito Circuits\ntranspiled to Quito"],
                                     x_limits=[0,13], y_limits=[0,25])
    #plotter.current_fig.suptitle("Transpiling_effects 5 qubits on backend " + backend)
    plotter.save_plots("BenchmarkComparisons/Plots")
    plt.show()
    return

if __name__ == "__main__":
    backend = "fake_quito"
    plot_results_transpiling(backend)