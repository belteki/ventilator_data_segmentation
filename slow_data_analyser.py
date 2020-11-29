# Import required packages
import IPython
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
import sys
import gc
import pickle
import zipfile
import scipy as sp
import sklearn as sk
from scipy import stats
from pandas import Series, DataFrame
from matplotlib import dates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from collections import defaultdict


# Set options
matplotlib.style.use('classic')
matplotlib.rcParams['figure.facecolor'] = 'w'


# Folders to import data from and export data to

# Folder on local computer to read the clinical data and blood gases from
DIR_READ_4 = os.path.join('/Users', 'guszti', 'ventilation_draeger')

# Folder on a USB stick to export data to
DATA_DUMP = os.path.join('/Volumes', 'Guszti', 'data_dump', 'draeger', 'analysis_individual')


# Create the folder if it does not exist
def create_data_dump_folder():
    # Images and raw data will be written on an external USB stick
    if not os.path.isdir(DATA_DUMP):
        os.makedirs(DATA_DUMP)


# Import anc process clinical details
def process_clinical_data(recording):
    # Import clinical details of all recordings
    clinical_details_old = pd.read_excel(os.path.join(DIR_READ_4, 'service_evaluation_patient_data.xlsx'))
    clinical_details_new = pd.read_excel(os.path.join(DIR_READ_4, 'ventilation_CO2_elimination_patient_data.xlsx'))                                     
    clinical_details = pd.concat([clinical_details_old, clinical_details_new])
    clinical_details.index = clinical_details['Recording']
    clinical_details['Recording start'] = pd.to_datetime(clinical_details['Recording period'].apply(lambda x: x[:10]),
                dayfirst = True)
    clinical_details['Recording end'] = pd.to_datetime(clinical_details['Recording period'].apply(lambda x: x[11:]),
                dayfirst = True)
    clinical_details['Postnatal age'] = clinical_details['Recording start'] - clinical_details['Date of birth']
    clinical_details['Corrected gestation'] = clinical_details['Gestation'] + \
        clinical_details['Postnatal age'].astype(int) / (1E+9 * 3600 * 24 * 7)
    clinical_details['Corrected gestation'] = round(clinical_details['Corrected gestation'], 2)
    clinical_details = clinical_details[clinical_details['Recording'] == recording]
    
    return clinical_details
    
    
# Import and process blood gases

def blood_gas_loader(path, dct, recording):
    '''
    str, dict, str -> None
    
    - path: filename with path of excel file containing the blood gases
    - dct: dictionary to contain blood gases;
    - recording: recording name

    Import blood gases as DataFrames into the dictionary 'dct'. This function modifies dct dictionary
    in place and return 'None'.

    '''
    dct[recording] = pd.read_excel(path, sheet_name = recording[:5], header = None)
    dct[recording] = pd.DataFrame(dct[recording].T)
    dct[recording].columns = dct[recording].iloc[0]
    dct[recording] = dct[recording][1:]
    dct[recording].index = [dct[recording]['Date:'], dct[recording]['Time:']]


def process_blood_gases(recording):
    blood_gases = {}

    if int(recording[2]) > 1:
        blood_gas_loader(os.path.join(DIR_READ_4, 'ventilation_CO2_elimination_blood_gases.xlsx'), 
                         blood_gases, recording) 
    
    else:
        blood_gas_loader(os.path.join(DIR_READ_4, 'service_evaluation_gases.xlsx'), blood_gases, recording) 

    pCO2s = blood_gases[recording][['pCO2, POC', 'Blood specimen type, POC']]

    # Change the index of pCO2s into single index format
    time_list_all = []    
    for i in range(len(pCO2s)):
        day = str(pCO2s.index[i][0])[:10]
        time = str(pCO2s.index[i][1])
        date_time = day + ' ' + time
        time_list_all.append(date_time)
    pCO2s.index = time_list_all

    # Convert the indices of the pCO2s DataFrames to datetime index
    pCO2s.index = pd.to_datetime(pCO2s.index)
    
    return pCO2s
    
    

# Functions to identify ventilator data files

def file_finder(recording):
    # Topic of the Notebook which will also be the name of the subfolder containing results
    TOPIC = 'analysis_individual'

    # Name of the external hard drive
    DRIVE = 'GUSZTI'

    # Folder on external drive to read the ventilation data from
    DIR_READ_1 =  os.path.join('/Volumes', DRIVE, 'Draeger', 'service_evaluation_old')
    DIR_READ_2 =  os.path.join('/Volumes', DRIVE, 'Draeger', 'service_evaluation_new')
    DIR_READ_3 =  os.path.join('/Volumes', DRIVE, 'Draeger', 'ventilation_CO2_elimination')

    x = int(recording.split('_')[0][-3:].lstrip('0'))
    
    if x <= 60:
        DIR_READ = DIR_READ_1
        
    elif 60 < x <= 111:
        DIR_READ = DIR_READ_2
    
    else:
        DIR_READ = DIR_READ_3

    flist = os.listdir(os.path.join(DIR_READ, recording))
    # There are some hidden files on the hard drive starting with '.' 
    # this step is necessary to ignore them
    
    flist = sorted(file for file in flist if not file.startswith('.'))
    
    return(DIR_READ, flist)


def data_finder(lst, tag):
    '''
    list -> list
    
    Takes a list of filenames and returns those ones that which contain 'tag'
    '''
    return [fname for fname in lst if tag in fname]
    
  

# Helper functions to import ventilator and process data
    
def data_loader_1(lst):
    
    '''
    list of string ->  list of DataFrames 
    
    - Takes a LIST OF STRINGS (csv files given as filenames with absolute path) and import them as a dataframes 
    - Combines the 'Date' and 'Time' columns into a Datetime index while keeping the original columns and
        makes it the index of the DataFrames
    - It returns a LIST OF DATAFRAMES. 
    
    '''
    data  = []
    for i, filename in enumerate(lst):
        # This is escaping characters with encoding errors with blackslashes that pd.csv can 
        # subsequently handle
        input_fd = open(lst[i], encoding='utf8' , errors = 'backslashreplace')
        frme = pd.read_csv(input_fd, keep_date_col = 'True', parse_dates = [['Date', 'Time']])
        input_fd.close()
        frme.index = frme.Date_Time
        data.append(frme)
    
    return data
    
    
    
def vent_mode_cleaner(inp):
    
    '''
    DataFrame -> DataFrame

    Takes a DataFrame of ventilator modes ('slow_text') and removes the unimmportant parameters and changes
    
    '''
    a = inp[inp.Id != 'Device is in neonatal mode']
    a = a[a.Id != 'Device is in neonatal mode']
    a = a[a.Id != 'Selected CO2 unit is mmHg']
    a = a[a.Id != "Selected language (variable text string transmitted with this code," + \
          " see 'MEDIBUS.X, Rules and Standards for Implementation' for further information)"]
    a = a[a.Id != 'Device configured for intubated patient ventilation']
    a = a[a.Id != 'Active humid heated']
    a = a[a.Id != 'Audio pause active']
    a = a[a.Id != 'Active humid unheated']
    a = a[a.Id != 'Tube type endotracheal']
    a = a[a.Id != 'Tracheal pressure calculation enabled as real-time value (independent of ATC adjunct)']
    a = a[a.Id != 'Suction maneuver active']
    
    return a    


    

def column_filter(col):
    
    '''
    list ->  list  
    
    Helper function used to filter which columns of the "slow_measurements" DataFrames are kept.
    It also modifies column names to make them simpler
    
    '''
    
    par_list = ['C20/Cdyn', 'Cdyn', 'DCO2', 'EIP', 'FiO2', 'MV', 'MVe', 'MVemand', 'MVespon', 
                'MVi', 'MVleak', 'PEEP', 'PIP', 'Pmean', 'Pmin', 'R', 'RR', 'RRmand', 'RRspon', 'TC',
                'TCe', 'Tispon', 'VTemand', 'VTespon', 'VThf', 'VTimand', 'VTispon', 'VTmand', 'VTspon',
                '?Phf', 'ΔPhf']
    
    if col in ['Time [ms]', 'Date', 'Time', 'Rel.Time [s]']:
        return col
      
    try:
        if col.split()[1] in ['leak', 'MVspon']:
            return col
    except IndexError:
        return
    
    if ' ' in col and col.split()[0][5:] in par_list:
        return col


def column_renamer(col):
    
    '''
    list ->  list  
    
    Removes pre_tags from columns.
    '''
    
    if col[:5] in ['5001|', '8272|']:
        return col[5:] 
    
    else:
        return col


def data_loader_2(lst):
    
    '''
    list of strings ->  list of DataFrames 
    
    - Takes a LIST OF STRINGS (csv files given as filenames with absolute path) and import them as a dataframes 
    - Combines the 'Date' and 'Time' columns into a Datetime index while keeping the original columns and
        makes it the index of the DataFrame
    - Appends the DataFrame to a list of DataFrames
    
    '''
    data  = []
    for i, filename in enumerate(lst):
        
        # This escaping characters with encoding errors with blackslashes that pd.csv can 
        # subsequently handle
        input_fd = open(lst[i], encoding='utf8' , errors = 'backslashreplace',)
        
        # Only returns selected columns
        frme = pd.read_csv(input_fd, keep_date_col = 'True', parse_dates = [['Date', 'Time']],
                          usecols = column_filter)
        input_fd.close()
        frme.index = frme.Date_Time
        
        # Filters and renames the columns using a helper function
        frme.rename(column_renamer, axis = 1, inplace = True)
        
        # Resampling to remove half-empty rows
        # This resampling works because the 'mean()' methods ignores na values as a default
        frme = frme.resample('1S').mean()

        data.append(frme) 
        
    return data




# Import and process ventilator modes
def process_vent_modes(recording):
    
    DIR_READ, flist = file_finder(recording)
    files = data_finder(flist, 'slow_Text')
    fnames = [os.path.join(DIR_READ, recording, filename) for filename in files]
    data = data_loader_1(fnames)
    vent_modes = {}
    
    for i, frme in enumerate(data):
        vent_modes[i] = data[i].reindex(['Id', 'Text', 'Rel.Time [s]'], axis = 1)
    
    for part in vent_modes:
        vent_modes[part] = vent_mode_cleaner(vent_modes[part])
        
    return vent_modes

#Import and process ventilator settings
def process_vent_settings(recording):

    parameters_to_keep = ['FiO2', 'VTi', 'Slope', 'Ti', 'Te', 'RR', 'Pinsp', 'PEEP', 
                          'Pmax', 'Flow trigger', 'MAPhf', 'VThf', 'Ampl hf', 
                          'Ampl hf max', 'fhf', 'ΔPsupp', ]
                          
    # `VTi` is not correctly labelled, it is actually the target expiratory tidal volume
    old_names = ['FiO2', 'VTi', 'Slope', 'Ti', 'Te', 'RR', 'Pinsp', 'PEEP', 'Pmax', 'Flow trigger',
                      'MAPhf', 'VThf', 'Ampl hf', 'Ampl hf max', 'fhf', 'ΔPsupp', ]

    new_names = ['FiO2_set', 'VTset', 'Slope', 'Ti_set', 'Te_set', 'RRset', 'PIP_set', 'PEEP_set', 
             'Pmax', 'Flow_trigger_set', 'MAPhf_set', 'VThf_set', 'Amplhf_set', 'Amplhf_max', 
              'fhf', 'ΔPsupp',]
              
    rename_dict = dict(zip(old_names, new_names))
    rename_dict

    to_weight_correct_settings = ['VTset', 'VThf_set']
    
    weight = process_clinical_data(recording)['Current weight'] / 1000 # in kilograms
    weight = weight.values[0]

    vent_settings = {}

    DIR_READ, flist = file_finder(recording)
    files = data_finder(flist, 'slow_Setting')
    fnames = [os.path.join(DIR_READ, recording, filename) for filename in files]
    
    data = data_loader_1(fnames)

    for i, frme in enumerate(data):
        vent_settings[i] = frme
        vent_settings[i] = vent_settings[i].reindex(
        ['Date_Time', 'Id', 'Unit', 'Value New', 'Value Old'], axis = 1)
    
    for part in vent_settings:
    
        # This is required as VTi is given both in mL and in L and keeping both would 
        # create duplicate columns during unstacking leading to an error message
        vent_settings[part] = vent_settings[part][vent_settings[part]['Unit'] != 'L'] 
        vent_settings[part] = vent_settings[part][vent_settings[part]['Id'].isin(parameters_to_keep)]
        vent_settings[part] = vent_settings[part].pivot('Date_Time', 'Id', 'Value New')
        vent_settings[part].fillna(method = 'ffill', inplace = True)
        vent_settings[part].rename(rename_dict, axis = 1, inplace = True)
        
        for par in to_weight_correct_settings:
            if par in vent_settings[part]:
                vent_settings[part][par] = vent_settings[part][par] / weight
        
    return vent_settings


# Import and process 1Hz ventilator data
def process_slow_measurements(recording):


    to_weight_correct_1 = ['MVe [L/min]', 'MVi [L/min]', 'MVespon [L/min]', 'MVemand [L/min]',
                       'MV [L/min]', 'MVleak [L/min]',
                       'VTmand [mL]', 'VTispon [mL]', 'VTemand [mL]', 'VTespon [mL]', 'VTimand [mL]',  
                       'VTspon [mL]', 'VThf [mL]']

    to_weight_correct_2 = ['DCO2 [10*mL^2/s]']

    weight = process_clinical_data(recording)['Current weight'] / 1000 # in kilograms
    weight = weight.values[0]
    
    old_names = ['% MVspon [%]', '% leak [%]', '?Phf [mbar]', 'C20/Cdyn [no unit]', 'Cdyn [L/bar]',
             'DCO2 [10*mL^2/s]', 'EIP [mbar]', 'FiO2 [%]', 'MV [L/min]', 'MVe [L/min]',
             'MVemand [L/min]', 'MVespon [L/min]', 'MVi [L/min]', 'MVleak [L/min]', 'PEEP [mbar]',
             'PIP [mbar]', 'Pmean [mbar]', 'Pmin [mbar]', 'R [mbar/L/s]', 'RR [1/min]',
             'RRmand [1/min]', 'RRspon [1/min]', 'Rel.Time [s]', 'TC [s]', 'TCe [s]', 'Time [ms]',
             'Tispon [s]', 'VTemand [mL]', 'VTespon [mL]', 'VThf [mL]', 'VTimand [mL]',
             'VTispon [mL]', 'VTmand [mL]', 'VTspon [mL]', 'ΔPhf [mbar]']

    new_names = ['MVspon_pc', 'Leak', 'Amplitude', 'C20/Cdyn', 'Cdyn',
             'DCO2', 'EIP', 'FiO2', 'MV', 'MVe',
             'MVemand', 'MVespon', 'MVi', 'MVleak', 'PEEP',
             'PIP', 'MAP', 'Pmin', 'R', 'RR',
             'RRmand', 'RRspon', 'Rel_time_s', 'TC', 'TCe', 'Time_ms',
             'Tispon', 'VTemand', 'VTespon', 'VThf', 'VTimand',
             'VTispon', 'VTmand', 'VTspon', 'Amplitude']

    rename_dict = dict(zip(old_names, new_names))
    
    slow_measurements = {}
    
    DIR_READ, flist = file_finder(recording)
    files = data_finder(flist, 'slow_Measurement')    
    fnames = [os.path.join(DIR_READ, recording, filename) for filename in files]
    
    data =  data_loader_2(fnames)

    for i, frme in enumerate(data):
        slow_measurements[i] = data[i]
        
    
    for part in slow_measurements:
    
        # Remove the duplicated and irrelevant VTmand [L] column
        for col in slow_measurements[part].columns:
            if col in ['VTmand [L]', 'VTspon [L]']:
                slow_measurements[part].drop(col, axis = 1, inplace = True)
                
        
        # Normalise parameters to body weight kg
        
        for par in to_weight_correct_1:
            if par in slow_measurements[part]:
                slow_measurements[part][par] = slow_measurements[part][par] / weight
    
        for par in to_weight_correct_2:
        # The original DCO2 values need to be multiplied by 10 as in the the downloaded data 
        # they are expressed as 1/10th of the DCO2 readings (see original column labels)
        # DCO2 needs to be weight corrected by the square of body weight
            if par in slow_measurements[part]:
                slow_measurements[part][par] = slow_measurements[part][par] * 10 / weight ** 2
        
        # Rename columns
        slow_measurements[part].rename(rename_dict, axis = 1, inplace = True)
        
        # Merge selected ventilator settings with selected ventilator parameters
        slow_measurements[part] = pd.merge(slow_measurements[part], 
        process_vent_settings(recording)[part], how = 'outer', left_index = True, right_index = True)
        slow_measurements[part].fillna(method = 'ffill', inplace = True)
        
        # Add Pinfl, VTdiff, Pdiff, RRdiff, VThf_diff
        if 'PIP' in slow_measurements[part] and 'PEEP_set' in slow_measurements[part]:
            slow_measurements[part]['Pinfl'] = slow_measurements[part]['PIP'] - \
                slow_measurements[part]['PEEP_set']
        
        if 'VTmand' in slow_measurements[part] and 'VTset' in slow_measurements[part]:   
            slow_measurements[part]['VTdiff'] = \
                slow_measurements[part]['VTmand'] - slow_measurements[part]['VTset']
    
        if 'Pmax' in slow_measurements[part] and 'PIP' in slow_measurements[part]:
            slow_measurements[part]['Pdiff'] = \
                slow_measurements[part]['Pmax'] - slow_measurements[part]['PIP']
    
        if 'RRmand' in slow_measurements[part] and 'RRset' in slow_measurements[part]:
            slow_measurements[part]['RRdiff'] = \
                slow_measurements[part]['RRmand'] - slow_measurements[part]['RRset']
        
        if 'VThf' in slow_measurements[part] and 'VThf_set' in slow_measurements[part]:
            slow_measurements[part]['VThf_diff'] = \
                slow_measurements[part]['VThf'] - slow_measurements[part]['VThf_set']
        
        if 'Amplitude' in slow_measurements[part] and 'Amplhf_set' in slow_measurements[part]:
            slow_measurements[part]['Ampl_diff'] = \
                slow_measurements[part]['Amplhf_set'] - slow_measurements[part]['Amplitude']
                
    return slow_measurements
    

def aggregator(slow_measurements):

    parametric_pars_all = {'MVspon_pc', 'Amplitude', 'C20/Cdyn', 'Cdyn', 'DCO2', 'EIP', 'MV', 'MVe',
                         'MVemand', 'MVespon', 'MVi',  'PEEP', 'PIP', 'MAP', 'Pmin', 'R', 
                  'RR', 'RRmand', 'RRspon', 'TC', 'TCe', 'Tispon', 'VTemand',
                  'VTespon', 'VThf', 'VTimand', 'VTispon', 'VTmand', 'VTspon', 'Pinfl'}

    nonparametric_pars_all = {'Leak', 'FiO2', 'MVleak', 'Pdiff', 'VTdiff', 'RRdiff', 'VThf_diff', 'Ampl_diff'}

    slow_measurements_1min_mean = {}
    slow_measurements_1min_median = {}
    slow_measurements_3h_rolling_mean = {}
    slow_measurements_3h_rolling_median = {}

    for part in slow_measurements:
    
        parametric_pars = parametric_pars_all & set(slow_measurements[part].columns)
        nonparametric_pars = nonparametric_pars_all & set(slow_measurements[part].columns)

        slow_measurements_1min_mean[part] = slow_measurements[part][parametric_pars].resample('1min').mean()
        slow_measurements_1min_median[part] = slow_measurements[part][nonparametric_pars].resample('1min').median()
    
        slow_measurements_3h_rolling_mean[part] = slow_measurements[part][parametric_pars].rolling(center = False, 
                                                                window = 10800 , min_periods=0).mean()
        slow_measurements_3h_rolling_median[part] = slow_measurements[part][nonparametric_pars].rolling(center = False, 
                                                                window = 10800 , min_periods=0).median()
    
    return (slow_measurements_1min_mean, slow_measurements_1min_median, 
            slow_measurements_3h_rolling_mean, slow_measurements_3h_rolling_median)
        
    
    
def frame_combiner(slow_measurements):
    slow_measurements_comb = pd.concat(slow_measurements, sort = False)
    slow_measurements_comb.index = slow_measurements_comb.index.droplevel(level = 0)
    return slow_measurements_comb
    

