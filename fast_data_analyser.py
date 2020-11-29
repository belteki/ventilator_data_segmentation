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



# Functions to identify ventilator data files

def file_finder(recording):
	
	# Name of the external hard drive where the files are kept
	DRIVE = 'Elements'

	# Folder on external drive to read the ventilation data from
	DIR_READ_1 =  os.path.join('/Volumes', DRIVE, 'Raw_data', 'Draeger', 
							   'service_evaluation_old')
	DIR_READ_2 =  os.path.join('/Volumes', DRIVE, 'Raw_data', 'Draeger', 
							   'service_evaluation_new')
	DIR_READ_3 =  os.path.join('/Volumes', DRIVE, 'Raw_data', 'Draeger', 
							   'ventilation_CO2_elimination')

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
	
  
  

# Helper function to import and process fast (100Hz) ventilator data	
def data_loader(lst, nrows):
	
	'''
	list of string ->  dictionary of DataFrames 
	
	- Takes a LIST OF STRINGS (csv files given as filenames with absolute path) and import them as a dataframes 
	- Combines the 'Date' and 'Time' columns into a Datetime index while keeping the original columns and
		makes it the index of the DataFrames
	- It returns a DICTIONARY OF DATAFRAMES. 
	
	'''
	data  = {}
	for filename in lst:
		# This is escaping characters with encoding errors with blackslashes that pd.csv can 
		# subsequently handle
		input_fd = open(filename, encoding='utf8' , errors = 'backslashreplace')
		frme = pd.read_csv(input_fd, usecols = [1, 2, 4, 5], keep_date_col = 'False', nrows = nrows,
					   parse_dates = [['Date', 'Time']])
		frme.index = frme.Date_Time
		frme = frme.iloc[:, [3,4]]
		frme.rename({'5001|Paw [mbar]' : 'pressure', '5001|Flow [L/min]' : 'flow'}, 
					axis = 'columns', inplace = True)				 
		input_fd.close()
		key = filename.split('_')[-4] + '_' + filename.split('_')[-3]
		data[key] = frme
		gc.collect()
	
	return data
	
	
	
# Import and process fast (100 Hz) ventilator data
def process_fast_data(recording, nrows = None):
	
	DIR_READ, flist = file_finder(recording)
	files = data_finder(flist, '_fast_Unknown.csv')
	fnames = [os.path.join(DIR_READ, recording, filename) for filename in files]
	data = data_loader(fnames, nrows)
		
	return data
	
	

# Helper function to import predicted_breaths (Ventiliser output)
def data_loader_2(lst):
	
	'''
	list of string ->  dictionary of DataFrames 
	
	- Takes a LIST OF STRINGS (csv files given as filenames with absolute path) and import them as a dataframes 
	- It returns a DICTIONARY OF DATAFRAMES. 
	
	'''
	data  = {}
	for filename in lst:
		
		input_fd = open(filename, encoding='utf8' , errors = 'backslashreplace')
		frme = pd.read_csv(input_fd)			 
		input_fd.close()
		key = filename.split('_')[-7] + '_' + filename.split('_')[-6]
		data[key] = frme
		gc.collect()
	
	return data


# Import predicted breaths (as reported by Ventiliser)
def process_predicted_breaths(recording):
	'''
	string -> dictionary of DataFrames
	'''
	DIR_READ, flist = file_finder(recording)
	files = data_finder(flist, '_predicted_Breaths_ms.csv')
	print(files)
	fnames = [os.path.join(DIR_READ, recording, filename) for filename in files]
	print(fnames)
	data = data_loader_2(fnames)
	
	return data
	
	


def flow_integrater(breaths):
	for breath in breaths.values():
		time_d_micros = pd.Series(np.zeros(len(breath)), index = breath.index)
		vol_d_ml = pd.Series(np.zeros(len(breath)), index = breath.index)
		for i in range(1, len(breath)):
			time_d_micros[i] = (breath.index[i] - breath.index[i-1]).microseconds
			vol_d_low = breath.iloc[i-1, 1] * (time_d_micros[i] / 60000 )
			vol_d_high = breath.iloc[i, 1] * (time_d_micros[i] / 60000 )
			vol_d_ml[i] = (vol_d_low + vol_d_high) / 2
		breath['volume'] = round(vol_d_ml.cumsum(), 4)

	return breaths



def PV_loop_all(recording, part, tag, pressures, volumes, pressure_high = None, vol_high = None, alpha = 0.02, 
                filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):
    
    fig, ax = plt.subplots(figsize  = [6, 6])
    ax.scatter(pressures, volumes, color = 'blue', s = 5, alpha = alpha )
    ax.set_xlim(0, pressure_high)
    ax.set_ylim(0, vol_high)
    ax.set_xlabel('Pressure (mbar)')
    ax.set_ylabel('Volume (mL)')
    ax.grid(True)

    if write: 
        fig.savefig('%s/%s_%s_period%s_PV_loop_all_alpha0%s.%s' % 
            (folder, recording, part[:-4], tag, str(alpha).split('.')[1], 'jpg'), dpi = dpi, 
            format = filetype, bbox_inches='tight', pad_inches=0.1)




def FV_loop_all(recording, part, tag, volumes, flows, vol_high=None, flow_low=None, flow_high=None, alpha = 0.02, 
                filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):
    
    fig, ax = plt.subplots(figsize  = [6, 6])
    ax.scatter(volumes, flows, color = 'blue', s = 5, alpha = alpha )
    ax.set_xlim(0, vol_high)
    ax.set_ylim(flow_low, flow_high)
    ax.set_xlabel('Volume (mL)')
    ax.set_ylabel('flow (L/min)')
    ax.grid(True)
    
    if write:
        
        fig.savefig('%s/%s_%s_period%s_FV_loop_all_alpha0%s.%s' % 
            (folder, recording, part[:-4], tag, str(alpha).split('.')[1], 'jpg'), dpi = dpi, 
            format = filetype, bbox_inches='tight', pad_inches=0.1)




def PV_loops_synchr_backup(recording, part, tag, pressures_synchr, volumes_synchr, pressures_backup, volumes_backup,
                           pressure_high = None, vol_high = None,
                           alpha_sync = 0.01, alpha_backup = 0.01, 
                           filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):
    
    fig, ax = plt.subplots(1, 2 , figsize  = [12, 6])
    ax[0].scatter(pressures_synchr, volumes_synchr, color = 'red', s = 5, alpha = alpha_sync )
    ax[1].scatter(pressures_backup, volumes_backup, color = 'blue', s = 5, alpha = alpha_backup )
    ax[0].set_xlim(0, pressure_high), ax[1].set_xlim(0, pressure_high)
    ax[0].set_ylim(0, vol_high), ax[1].set_ylim(0, vol_high)
    ax[0].set_xlabel('Pressure (mbar)'), ax[1].set_xlabel('Pressure (mbar)')
    ax[0].set_ylabel('Volume (mL)'), ax[1].set_ylabel('Volume (mL)')
    ax[0].set_title('Synchronised inflations'), ax[1].set_title('Backup inflations')
    ax[0].grid(True), ax[1].grid(True)

    if write:
    
        fig.savefig('%s/%s_%s_period%s_PV_loop_all_synch_backup_alpha_sync0%s_alpha_backup0%s.%s' % 
            (folder, recording, part[:-4], tag, 
             str(alpha_sync).split('.')[1], str(alpha_backup).split('.')[1], 'jpg'), 
             dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1)




def PV_loops_synchr_backup_spont(recording, part, tag, pressures_synchr, volumes_synchr, pressures_backup, volumes_backup,
                                 pressures_spont, volumes_spont,
                                 pressure_high = None, vol_high = None,
                                 alpha_sync = 0.01, alpha_backup = 0.01, alpha_spont = 0.01, 
                                 filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):
    
    fig, ax = plt.subplots(1, 3 , figsize  = [18, 6])
    ax[0].scatter(pressures_synchr, volumes_synchr, color = 'red', s = 5, alpha = alpha_sync )
    ax[1].scatter(pressures_backup, volumes_backup, color = 'blue', s = 5, alpha = alpha_backup )
    ax[2].scatter(pressures_spont, volumes_spont, color = 'blue', s = 5, alpha = alpha_spont )
    
    for i in range(3):
        ax[i].set_xlim(0, pressure_high)
        ax[i].set_ylim(0, vol_high)
        ax[i].set_xlabel('Pressure (mbar)')
        ax[i].set_ylabel('Volume (mL)')
        ax[i].grid(True), 
        
    ax[0].set_title('Synchronised inflations')
    ax[1].set_title('Backup inflations')
    ax[2].set_title('Spontaneous breaths')
    
    if write:
    
        fig.savefig('%s/%s_%s_period%s_PV_loop_all_synch_backup_spont_alpha_sync0%s_alpha_backup0%s.%s' % 
            (folder, recording, part[:-4], tag, 
             str(alpha_sync).split('.')[1], str(alpha_backup).split('.')[1], 'jpg'), 
             dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1)




def FV_loops_synchr_backup(recording, part, tag, volumes_synchr, flows_synchr, volumes_backup, flows_backup,
                           vol_high, flow_low, flow_high, alpha_sync = 0.02, alpha_backup = 0.02,
                           filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):
    
    fig, ax = plt.subplots(1, 2 , figsize  = [12, 6])
    ax[0].scatter(volumes_synchr, flows_synchr, color = 'red', s = 5, alpha = alpha_sync)
    ax[1].scatter(volumes_backup, flows_backup, color = 'blue', s = 5, alpha = alpha_backup)
    ax[0].set_xlim(0, vol_high), ax[1].set_xlim(0, vol_high)
    ax[0].set_ylim(flow_low, flow_high), ax[1].set_ylim(flow_low, flow_high)
    ax[0].set_xlabel('Volume (mL)'), ax[1].set_xlabel('Volume (mL)')
    ax[0].set_ylabel('Flow (L/min)'), ax[1].set_ylabel('Flow (L/min)')
    ax[0].set_title('Synchronised inflations'), ax[1].set_title('Backup inflations')
    ax[0].grid(True), ax[1].grid(True)

    if write:
        fig.savefig('%s/%s_%s_period%s_FV_loop_all_synch_backup_alpha_sync0%s_alpha_backup0%s.%s' % 
            (folder, recording, part[:-4], tag, 
             str(alpha_sync).split('.')[1], str(alpha_backup).split('.')[1], 'jpg'),
            dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1)




def FV_loops_synchr_backup_spont(recording, part, tag, volumes_synchr, flows_synchr, volumes_backup, flows_backup,
                                 volumes_spont, flows_spont,
                                 vol_high, flow_low, flow_high, alpha_sync = 0.02, alpha_backup = 0.02,
                                 alpha_spont = 0.02,
                                 filetype = 'jpg', dpi = 200, write = True, folder = os.getcwd()):

    fig, ax = plt.subplots(1, 3 , figsize  = [18, 6])
    ax[0].scatter(volumes_synchr, flows_synchr, color = 'red', s = 5, alpha = alpha_sync)
    ax[1].scatter(volumes_backup, flows_backup, color = 'blue', s = 5, alpha = alpha_backup)
    ax[2].scatter(volumes_spont, flows_spont, color = 'blue', s = 5, alpha = alpha_spont)
    
    for i in range(3):
        ax[i].set_xlim(0, vol_high)
        ax[i].set_ylim(flow_low, flow_high)
        ax[i].set_xlabel('Volume (mL)')
        ax[i].set_ylabel('Flow (L/min)')
        ax[i].grid(True)
    
    ax[0].set_title('Synchronised inflations')
    ax[1].set_title('Backup inflations')
    ax[2].set_title('Spontaneous breaths')

    if write:
        fig.savefig('%s/%s_%s_period%s_FV_loop_all_synch_backup_spont_alpha_sync0%s_alpha_backup0%s.%s' % 
            (folder, recording, part[:-4], tag, 
             str(alpha_sync).split('.')[1], str(alpha_backup).split('.')[1], 'jpg'),
            dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1)




def wave_individual(recording, part, breath, i, dpi = 200, filetype = 'jpg', show = True, 
                    write = False, folder = os.getcwd()):
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex = 'all', sharey = 'none')
    fig.set_size_inches(12,9); fig.set_label('res')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
    
    breath.pressure.plot(ax = axes[0], color = 'red', title = 'Pressure', linewidth=2, 
            ylim = [0, (breath.pressure.max() * 1.2)], x_compat = True);
    breath.flow.plot(ax = axes[1], color = 'green', title = 'Flow', linewidth=2,
            ylim = [(breath.flow.min() * 1.2), (breath.flow.max() * 1.2)], x_compat = True)
    xmin, xmax = axes[1].get_xlim()
    axes[1].hlines(0, xmin, xmax, color = 'black', linewidth = 2)
    breath.volume.plot(ax = axes[2], color = 'blue', title = 'Volume', linewidth=2, 
            ylim = [-0.1, (breath.volume.max() * 1.2)], x_compat=True)
    
    majorFmt = dates.DateFormatter('%H:%M:%S.%f')  
    axes[2].xaxis.set_major_formatter(majorFmt)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0, fontsize = 12, 
             horizontalalignment = 'left')
    
    axes[0].set_xlabel(''); axes[1].set_xlabel('')
    axes[2].set_xlabel('Time', size = 16, color = 'black', rotation = 0 )
    axes[0].set_ylabel('mbar', size = 16, color = 'black')
    axes[1].set_ylabel('L/min', size = 16, color = 'black')
    axes[2].set_ylabel('mL', size = 16, color = 'black')
    axes[0].set_title('Pressure', size = 20, color = 'black')
    axes[1].set_title('Flow', size = 20, color = 'black')
    axes[2].set_title('Volume', size = 20, color = 'black')

    axes[0].grid('on', linestyle='-', linewidth=0.5, color = 'gray') 
    axes[1].grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    axes[2].grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    
    if write:
        fig.savefig('%s/%s_%s_breath%d_wave.%s' % 
            (folder, recording, part[:-4], i, 'jpg'), dpi = dpi, format = filetype,
            bbox_inches='tight', pad_inches=0.1,)
        
    if not show:
        plt.close()



def loops(recording, part, breath, i, dpi = 200, filetype = 'jpg', show = True, write = False, folder = os.getcwd()):
    
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(12,6)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    
    x = breath.pressure
    y = breath.flow
    z = breath.volume

    ax0.plot(x, z, linewidth = 2, color = 'red',)
    ax0.set(xlim = [0, (x.max() * 1.2)], ylim = [0, (z.max() * 1.2)])

    ax0.set_title('Pressure - Volume loop', size = 20, color = 'black')
    ax0.set_xlabel('Pressure (mbar)', size = 16, color = 'black')
    ax0.set_ylabel('Volume (mL)', size = 16, color = 'black')
    ax0.grid('on', linestyle='-', linewidth=0.5, color = 'gray')
  
    ax1.plot(y, z, linewidth = 2, color = 'blue',)
    ax1.set(xlim = [y.min() * 1.2, (y.max() * 1.2)], ylim = [0, (z.max() * 1.2)])
    
    ax1.set_title('Flow - Volume loop', size = 20, color = 'black')
    ax1.set_xlabel('Flow (L/min)', size = 16, color = 'black')
    ax1.set_ylabel('Volume (mL)', size = 16, color = 'black')
    ax1.grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    
    if write:
        fig.savefig('%s/%s_%s_breath%d_loop.%s' % 
        (folder, recording, part[:-4], i, 'jpg'), dpi = dpi, format = filetype,
         bbox_inches='tight', pad_inches=0.1,)
        
    if not show:
        plt.close()



def waves_multiple(recording, part, breaths, dpi = 200, filetype = 'jpg', show = True, write = False, folder = os.getcwd()):
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex = 'all', sharey = 'none')
    fig.set_size_inches(21,15); fig.set_label('res')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
    
    breaths.pressure.plot(ax = axes[0], color = 'red', title = 'Pressure', linewidth=2, 
                ylim = [0, (breaths.pressure.max() * 1.2)]);
    breaths.flow.plot(ax = axes[1], color = 'green', title = 'Flow', linewidth=2,
                ylim = [(breaths.flow.min() * 1.2), (breaths.flow.max() * 1.2)])
    breaths.volume.plot(ax = axes[2], color = 'blue', title = 'Volume', linewidth=2,
                ylim = [0, (breaths.volume.max() * 1.2)], x_compat=True)
    xmin, xmax = axes[1].get_xlim()
    axes[1].hlines(0, xmin, xmax, color = 'black', linewidth = 2)
    
    majorFmt = dates.DateFormatter('%H:%M:%S\n%d/%m/%y')  
    axes[2].xaxis.set_major_formatter(majorFmt)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0, fontsize = 20, 
             horizontalalignment = 'center')
      
    axes[0].set_xlabel(''); axes[1].set_xlabel('')
    axes[2].set_xlabel('Time', size = 20, color = 'black', rotation = 0 )
    axes[0].set_ylabel('mbar', size = 20, color = 'black')
    axes[1].set_ylabel('L/min', size = 20, color = 'black')
    axes[2].set_ylabel('mL', size = 20, color = 'black')
    axes[0].set_title('Pressure', size = 20, color = 'black')
    axes[1].set_title('Flow', size = 20, color = 'black')
    axes[2].set_title('Volume', size = 20, color = 'black')
    
    axes[0].grid('on', linestyle='-', linewidth=0.5, color = 'gray') 
    axes[1].grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    axes[2].grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    
    if write:
        fig.savefig('%s/%s_%s_breaths%d_%d_wave.%s' % 
            (folder, recording, part[:-4], breaths['level_0'][0], breaths['level_0'][-1], 'jpg'), 
            dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1,)
        
    if not show:
        plt.close()



def loops_multiple(recording, part, breaths, dpi = 200, filetype = 'jpg', show = True, write = False, folder = os.getcwd()):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(12,6)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    
    x = breaths.pressure
    y = breaths.flow
    z = breaths.volume

    ax0.plot(x, z, linewidth = 2, color = 'red')
    ax0.set(xlim = [0, (x.max() * 1.2)], ylim = [0, (z.max() * 1.2)])

    ax0.set_title('Pressure - Volume loop', size = 20, color = 'black')
    ax0.set_xlabel('Pressure (mbar)', size = 16, color = 'black')
    ax0.set_ylabel('Volume (mL)', size = 16, color = 'black')
    ax0.grid('on', linestyle='-', linewidth=0.5, color = 'gray')
  
    ax1.plot(y, z, linewidth = 2, color = 'blue')
    ax1.set(xlim = [y.min() * 1.2, (y.max() * 1.2)], ylim = [0, (z.max() * 1.2)])
    
    ax1.set_title('Flow - Volume loop', size = 20, color = 'black')
    ax1.set_xlabel('Flow (L/min)', size = 16, color = 'black')
    ax1.set_ylabel('Volume (mL)', size = 16, color = 'black')
    ax1.grid('on', linestyle='-', linewidth=0.5, color = 'gray')
    
    if write:
        fig.savefig('%s/%s_%s_breaths%d_%d_loop.%s' % 
        (folder, recording, part[:-4], breaths['level_0'][0], breaths['level_0'][-1], 'jpg'),
         dpi = dpi, format = filetype, bbox_inches='tight', pad_inches=0.1,)
        
    if not show:
        plt.close()




