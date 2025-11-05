import os
import matplotlib.pyplot as plt
import numpy as np

class DataPlotter:
    def __init__(self, figsize=[10, 5]):
        self.plots = {}
        plt.rcParams['figure.figsize'] = figsize
        return
    
    def addfig(self, name, nrows=1, ncols=1, addtitle=None, *args, **kwargs):
        self.current_fig, self.current_ax = plt.subplots(nrows=nrows, ncols=ncols,
                                                         *args, **kwargs)
        self.current_name = name
        self.plots[name] = [self.current_fig, self.current_ax]
        if addtitle is False:
            pass
        elif addtitle is None:
            self.current_fig.suptitle(name)
        else:
            self.current_fig.suptitle(addtitle)
        return
    
    def set_current_fig(self, name):
        [self.current_fig, self.current_ax] = self.plots[name]
        return
    
    def add_multiplot(self, ydata, xdata=None, plottype="plot", labels=None,
                      **kwargs):
        self.current_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if type(ydata) is np.ndarray:
            self.addplot(ydata, xdata, plottype=plottype, **kwargs) 
        elif labels is None:
            if type(xdata) is np.ndarray:
                for y in ydata:
                    self.addplot(y, xdata, plottype=plottype, **kwargs)    
            else:
                for x, y in zip(xdata, ydata):
                    self.addplot(y, x, plottype=plottype, **kwargs)
        else:
            if type(xdata) is np.ndarray:
                for y, label in zip(ydata, labels):
                    self.addplot(y, xdata, plottype=plottype, label=label, 
                                 **kwargs)    
            else:
                for x, y, label in zip(xdata, ydata, labels):
                    self.addplot(y, x, plottype=plottype, label=label,
                                 **kwargs)

        return

    def addplot(self, ydata, xdata=None, plottype="plot", **kwargs):
        if xdata is None:
            xdata = np.arange(len(ydata))
        plot_function = getattr(self.current_ax, plottype, None)
        if plot_function is None:
            raise ValueError("plottype unknown")
        plot_function(xdata, ydata, **kwargs)
        return
    
    def multi_axis_plot(self, titles, ydata, xdata=None, plottype="plot",
                        **kwargs):
        #function only works for a single row of specified x values
        if len(titles) == 1:
            if xdata is not None and len(xdata) == 1:
                xdata = xdata[0]
            if len(ydata) == 1:
                ydata = ydata[0]
            if xdata is None or xdata.shape != ydata.shape:
                N_ydata = ydata.shape[0]
                for k in range(N_ydata):
                    self.addplot(ydata[k,:], xdata=xdata, plottype=plottype, **kwargs)
            else:
                self.addplot(ydata, xdata=xdata, plottype=plottype, **kwargs)
            if type(titles) is not str:
                plot_title_str = str(titles)
            self.current_ax.set_title(plot_title_str)
        else:
            if type(ydata) is np.ndarray:
                data_rows = len(titles)
                ydata = [ydata[a,:] for a in range(data_rows)]
            for a, (plot_title, plot_data) in enumerate(zip(titles, ydata)):
                if type(plot_title) is not str:
                    plot_title_str = str(plot_title)
                else:
                    plot_title_str = plot_title
                if xdata is None:
                    xdata = np.arange(len(plot_data))
                plot_function = getattr(self.current_ax[a], plottype, None)
                if xdata.shape != plot_data.shape:
                    N_ydata = plot_data.shape[0]
                    for k in range(N_ydata):
                        plot_function(xdata, plot_data[k,:], **kwargs)
                else:
                    plot_function(xdata, plot_data, **kwargs)
                self.current_ax[a].set_title(plot_title_str)
        return

    def multi_axis_plots_add_labels(self, labels):
        self.current_ax[-1].legend(labels)

    def show(self):
        #plt.tight_layout()
        plt.show()
        return
    
    def save_plots(self, folder):
        for name, (fig, ax) in self.plots.items():
            file_str = name + ".png"
            fig.savefig(os.path.join(folder, file_str), bbox_inches='tight', dpi=600)
        return
    
    def save_currentfig(self, folder, name=None):
        if name is None:
            name = self.current_name
        file_str = name + ".png"
        self.current_fig.savefig(os.path.join(folder, file_str), bbox_inches='tight', dpi=600)
        return

