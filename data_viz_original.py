import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from sklearn.metrics import r2_score
from tqdm import tqdm
import peakutils

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

#We want to get the index when the voltage last reaches its peak
def get_last_max_voltage_ind(df):
    return df.loc[df['Voltage (V)']<-0.5, 'TIME (ms)'].index[-1] #the -0.5 is bound to change depending on the files we're looking at

#Since nothing interesting happens long after the pulse, we can remove time after 1500ms
def remove_late_time(df, thres, min_dist):
    _, _, peak_ind = get_peaks(df, thres, min_dist)
    last_peak_ind = peak_ind[-1]
    df = df.drop(df.index[last_peak_ind+120:])
    return df 

#All the previous manipulation of the time has mixed up the indexes, so we need to reset them
def res_index(df):
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df

#Some of the data was measured with an inverted current probe so we need to compensate by *-1
def invert_positive_current(df):
    if abs(max(df["Current (mA)"]) - df["Current (mA)"][0]) > abs(min(df["Current (mA)"]) - df["Current (mA)"][0]):
        df["Current (mA)"] = df["Current (mA)"]*(-1)
    return df 

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

#This is everything we do to the dataframe 
def do_all_to_df(file):
    df = pd.read_csv(file)
    df = invert_positive_current(df)
    return df

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

#This is the model for the approximation
def f(x:list, y0, A ,t1):
    return y0 + A*np.exp(-x/t1)


#This is a function that will return the x_values and y_values for the peaks of the curve
def get_peaks(df, thres=0.8, min_dist=100):
    peak_indexes = peakutils.indexes(-df["Current (mA)"], thres, min_dist)
    peak_indexes = [int(peak) for peak in peak_indexes]
    peak_x = df["TIME (ms)"].values[peak_indexes]
    peak_y = df["Current (mA)"].values[peak_indexes]
    return peak_x, peak_y, peak_indexes


#This function is to create the interval on which we want to do the approximation 
#It needs to be changed to account for what we consider the maximum current (is it positive or negative)
#1000 because we look at the different graphs between 0 and 1500 
def trim_for_fitting(df, thres, min_dist, max_length=50):
    _, _, peak_ind = get_peaks(df, thres, min_dist)
    last_peak_ind = peak_ind[-1]
    last_index = df.index[-1]
    if last_index - last_peak_ind > max_length:
        return df[last_peak_ind:last_peak_ind+max_length]
    else:
        return df[last_peak_ind:last_index]

# In case of an s file current reaches saturation and oscillates around it so the peak is not a good indicator for the beginning of the fit 
def trim_for_fitting_s(df, max_length=50):
    last_vol_ind = get_last_max_voltage_ind(df)
    if df.index[-1]-last_vol_ind>max_length:
        return df[last_vol_ind:last_vol_ind+max_length]
    else:
        return df[last_vol_ind:df.index[-1]]


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

import re
def parse_name(time_str):
    # Extract the numeric value and the unit of time from the string
    value, unit = re.match(r'([\d.]+)([a-z]+)?', time_str).groups()
    value = float(value)
    
    # Convert the value to seconds
    if unit is None:
        return value
    elif unit == 's':
        return value
    elif unit == 'ms':
        return value / 1000
    elif unit == 'p':
        return value
    elif unit == 'k':
        return value * 1000
    elif unit == 'm':
        return value * 1000 * 1000
    else:
        raise ValueError(f'Invalid time unit: {unit}')
        
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# This function is for everything but original files
# string_ending is :
# 2p.csv for 2p files
# p.csv for p files
# hz.csv for hz files
# .csv for s files
def parse_dataframe(df, string_ending):
    parsed_dataframe = df
    parsed_dataframe['sortable_name'] = parsed_dataframe['name'].str.replace(string_ending, '').str.replace('tsf', '')
    parsed_dataframe['sortable_name'] = parsed_dataframe['sortable_name'].apply(parse_name)    
    parsed_dataframe = parsed_dataframe.sort_values(by=['sortable_name'])
    return parsed_dataframe

# This function removes files with a r2 value below a certain threshold
def filter_dataframe(df, thres=0.5):
    filtered_dataframe = df[df['r2'] >= thres]
    return filtered_dataframe

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


# class for a single file 
class OriginalFile:
    def __init__(self, filepath, show_fit=True):
        self.filepath = filepath
        self.show_fit = show_fit
        self.name = filepath.split('\\')[-1]
        self.df = do_all_to_df(filepath)
        self.model_function = f
        if self.show_fit:
            if self.name.endswith("2p.csv"):
                self.df = remove_late_time(self.df, thres=0.6, min_dist=50)
                self.fitting_df = trim_for_fitting(self.df, thres=0.6, min_dist=50, max_length=50)
                self.popt, _ = optimize.curve_fit(self.model_function, self.fitting_df["TIME (ms)"], self.fitting_df["Current (mA)"], p0=[-0.5, 10^8, 500], maxfev=30000)
            elif self.name.endswith("p.csv"):
                self.df = remove_late_time(self.df, thres=0.4, min_dist=1)
                self.fitting_df = trim_for_fitting(self.df, thres=0.4, min_dist=1, max_length=50)
                self.popt, _ = optimize.curve_fit(self.model_function, self.fitting_df["TIME (ms)"], self.fitting_df["Current (mA)"], p0=[-0.5, 10^8, 500], maxfev=30000)
            elif self.name.endswith("hz.csv"):
                self.df = remove_late_time(self.df, thres=0.5, min_dist=10)
                self.fitting_df = trim_for_fitting(self.df, thres=0.6, min_dist=1, max_length=50)
                self.popt, _ = optimize.curve_fit(self.model_function, self.fitting_df["TIME (ms)"], self.fitting_df["Current (mA)"], p0=[-0.3, 10^8, 1000], maxfev=120000)
            elif self.name.endswith("s.csv"):
                self.fitting_df = trim_for_fitting_s(self.df, max_length=50)
                self.popt, _ = optimize.curve_fit(self.model_function, self.fitting_df["TIME (ms)"], self.fitting_df["Current (mA)"], p0=[2, 10^8, 500], maxfev=60000)
            else:
                self.df = self.df[self.df['TIME (ms)'] <= 1500]
                self.fitting_df = trim_for_fitting(self.df, thres=0.6, min_dist=50, max_length=50)
                self.popt, _ = optimize.curve_fit(self.model_function, self.fitting_df["TIME (ms)"], self.fitting_df["Current (mA)"], p0=[-0.8, 100, 50], maxfev=30000)

            self.t1, self.r2 = self.popt[2], r2_score(self.fitting_df["Current (mA)"], f(self.fitting_df["TIME (ms)"], *self.popt))
    
    # A function that takes in self.df and show_fit=True and plots the Current and Voltage against TIME and shows the fitted curve if show_fit is True
    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(self.df["TIME (ms)"], self.df["Current (mA)"])
        ax.set_title(self.name)
        ax.set_ylabel('Current in mA')
        ax.set_xlabel('Time in ms')   
        ax.invert_yaxis()
        ax2 = ax.twinx()
        ax2.plot(self.df["TIME (ms)"], self.df["Voltage (V)"], color='#00ff00', linestyle='dashed', alpha=0.4)
        ax2.set_ylabel('Voltage in V')
        ax2.invert_yaxis()
        if self.show_fit:
            ax.plot(self.fitting_df["TIME (ms)"], self.model_function(self.fitting_df["TIME (ms)"], *self.popt), color='red')
            textbox = dict(boxstyle='round', facecolor='yellow')
            ax.text(0.35, 0.97, f'value of t1 :{round(self.t1, 2)} ms\nR^2={round(self.r2, 3)}',
            transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=textbox)
        fig.tight_layout()
        plt.show()

#---------------------------------------------------

# Class for several files
class MultipleOriginalFiles:
    def __init__(self, original_files):
        self.original_files = original_files
        self.dataframe = pd.DataFrame(columns=['name', 't1', 'r2'])
        for original_file in self.original_files:
            if original_file.show_fit:
                new_line = {'name': original_file.name, 't1': original_file.t1, 'r2': original_file.r2}      
                self.dataframe = self.dataframe.append(new_line, ignore_index=True)

    def plot(self):
        fig, ax = plt.subplots(len(self.original_files)//2 + len(self.original_files)%2, 2, figsize=(16, 9))
        pbar = tqdm(total=len(self.original_files))
        for i, original_file in enumerate(self.original_files):
            # Plotting Current and Voltage against time
            ax.flat[i].plot(original_file.df["TIME (ms)"], original_file.df["Current (mA)"])
            ax.flat[i].set_title(original_file.name)
            ax.flat[i].set_ylabel('Current in mA')
            ax.flat[i].set_xlabel('Time in ms')
            ax.flat[i].invert_yaxis()
            ax2 = ax.flat[i].twinx()
            ax2.plot(original_file.df["TIME (ms)"], original_file.df["Voltage (V)"], color='#00ff00', linestyle='dashed', alpha=0.4)
            ax2.set_ylabel('Voltage in V')
            ax2.invert_yaxis()
            if original_file.show_fit:
                # We can only add the t1 and r2 values to the dataframe if the fit is shown
                self.dataframe = self.dataframe.append({'name': original_file.name, 't1': original_file.t1, 'r2': original_file.r2}, ignore_index=True)      
                # Plotting the graph and the textbox
                ax.flat[i].plot(original_file.fitting_df["TIME (ms)"], original_file.model_function(original_file.fitting_df["TIME (ms)"], *original_file.popt), color='red')
                textbox = dict(boxstyle='round', facecolor='yellow')
                ax.flat[i].text(0.35, 0.97, f'value of t1 :{round(original_file.t1, 2)} ms\nR^2={round(original_file.r2, 3)}',
                transform=ax.flat[i].transAxes, fontsize=9,verticalalignment='top', bbox=textbox)
            pbar.update(1)
        fig.tight_layout()
        plt.show()
    
    # This function plots the final results of the t1 values, it doesn't work for the original files
    def plot_effect_on_t1(self, title, string_ending, thres=0.6):
        if string_ending == '2p.csv':
            xlabel = 'Distance between pulses in s'
        elif string_ending == 'p.csv':
            xlabel = 'Number of pulses'
        elif string_ending == 'hz.csv':
            xlabel = 'Frequency in hz'
        elif string_ending == '.csv':
            xlabel = 'Pulse duration in s'
        fil_par_dataframe = filter_dataframe(parse_dataframe(self.dataframe, string_ending), thres)
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(fil_par_dataframe['name'], fil_par_dataframe['t1'], marker='x')
        ax.set_title(title)
        ax.set_ylabel('t1 in ms')
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        plt.show()


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def make_object_list(folder, list_of_files, show_fit):
    list_of_objects = []
    pbar = tqdm(total=len(list_of_files))
    for file in list_of_files:
        list_of_objects.append(OriginalFile(folder + '\\' + file, show_fit))
        pbar.update(1)
    return list_of_objects



import os
if __name__ == '__main__':
    two_p_csv = []
    p_csv = []
    hz_csv = []
    s_csv = []
    other = []
    folder = '.\\low_MW'
    # Iterate through all files in the directory
    for filename in os.listdir(folder):
        # Check the file extension
        if filename.endswith("2p.csv"):
            two_p_csv.append(filename)
        elif filename.endswith("p.csv"):
            p_csv.append(filename)
        elif filename.endswith("hz.csv"):
            hz_csv.append(filename)
        elif filename.endswith("s.csv"):
            s_csv.append(filename)
        else:
            other.append(filename)
    hz_csv = ['tsf1hz.csv', 'tsf5hz.csv']
    hz_obj_list = make_object_list(folder, hz_csv, True)
    mul_hz_obj = MultipleOriginalFiles(hz_obj_list)
    # mul_hz_obj.plot()
    mul_hz_obj.plot_effect_on_t1('Effect of frequency on t1', 'hz.csv', 0.6)



    

