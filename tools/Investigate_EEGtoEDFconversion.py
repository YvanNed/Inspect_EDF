# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 18:06:46 2025

@author: yvan.nedelec
"""

# trying to understand the problem of converting BrainVision data to .edf format
# when we manually set the dynamic range to convert in .edf, some channels appears to be flat (while they were not as .eeg file)

#%% import
import os
import re
import mne
import yasa
import glob
import chardet
import sklearn
import pyedflib
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt

# %matplotlib qt

# custom function to detect automatically and return the encoding of edf file
def detect_encoding(byte_string, min_confidence=0.6):
    result = chardet.detect(byte_string)
    encoding = result['encoding']
    confidence = result['confidence']
    if encoding is None or confidence < min_confidence:
        raise UnicodeDecodeError("chardet", byte_string, 0, len(byte_string),
                                 f"\tUnable to reliably detect encoding. Detected: {encoding} with confidence {confidence}")
    return encoding

# custom function to read information from EDF headers, without using the pyedflib package (that was too strict for ICEBERG)
# EDF file should follow a strict format, dedicating a specific number of octets for each type of information.
# it means that we can read the info octet by octet by specifying the number of octets we expect for the next variable (that is known from the EDF norm)
def read_edf_header_custom(file_path):
    with open(file_path, 'rb') as f: # open the file in binary mode, to read octet by octet. 
        header = {}
        # detect encoding
        raw_header = f.read(256)
        encoding = detect_encoding(raw_header)
        # print(f"\tDetected encoding for {file_path} : {encoding}")
        # Rewind to the beginning of the file
        f.seek(0)
        
        # the first 256 octets are global subject info
        header['version'] = f.read(8).decode(encoding).strip()
        header['patient_id'] = f.read(80).decode(encoding).strip()
        header['recording_id'] = f.read(80).decode(encoding).strip()
        header['start_date'] = f.read(8).decode(encoding).strip()
        header['start_time'] = f.read(8).decode(encoding).strip()
        header['header_bytes'] = int(f.read(8).decode(encoding).strip())
        header['reserved'] = f.read(44).decode(encoding).strip()
        header['n_data_records'] = int(f.read(8).decode(encoding).strip())
        header['duration_data_record'] = float(f.read(8).decode(encoding).strip())
        header['n_channels'] = int(f.read(4).decode(encoding).strip())
        
        # get info per channel
        n = header['n_channels']
        channel_fields = {
            'channel': [],
            'transducer_type': [],
            'dimension': [],
            'physical_min': [],
            'physical_max': [],
            'digital_min': [],
            'digital_max': [],
            'prefiltering': [],
            'sampling_frequency': [],
            'reserved': [],
        }

        for key in channel_fields:
            length = {
                'channel': 16,
                'transducer_type': 80,
                'dimension': 8,
                'physical_min': 8,
                'physical_max': 8,
                'digital_min': 8,
                'digital_max': 8,
                'prefiltering': 80,
                'sampling_frequency': 8,
                'reserved': 32,
            }[key]
            channel_fields[key] = [f.read(length).decode(encoding).strip() for _ in range(n)]

        header.update(channel_fields)
    
    return header

from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_mne_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, 
                  overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    update 2021: edf export is now also supported in MNE:
    https://mne.tools/stable/generated/mne.export.export_raw.html
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    print('did you know EDF export is now supported in MNE via edfio? have a look at https://mne.tools/stable/generated/mne.export.export_raw.html')
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
        
    # static settings
    has_annotations = True if len(mne_raw.annotations)>0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767 
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607
    
    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])
    
    if tmin:
        date += timedelta(seconds=tmin)
    # no conversion necessary, as pyedflib can handle datetime.
    #date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)
    
    # define a fixed dynamic range of 1000 µV
    EEG_SPAN = 1000.0  # ±500 µV
    phys_min = -EEG_SPAN / 2
    phys_max =  EEG_SPAN / 2
    
    # create channel from this   
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels, 
                               file_type=file_type)
        
        channel_info = []
        
        ch_idx = range(n_channels) if picks is None else picks
        keys = list(mne_raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {'label': mne_raw.ch_names[i], 
                           'dimension': 'uV', 
                           'sample_frequency': mne_raw._raw_extras[0]['n_samps'][i], 
                           # 'physical_min': mne_raw._raw_extras[0]['physical_min'][i], 
                           'physical_min': float(phys_min), 
                           # 'physical_max': mne_raw._raw_extras[0]['physical_max'][i], 
                           'physical_max': float(phys_max), 
                           'digital_min':  mne_raw._raw_extras[0]['digital_min'][i], 
                           'digital_max':  mne_raw._raw_extras[0]['digital_max'][i], 
                           'transducer': '', 
                           'prefilter': ''}
            except:
                ch_dict = {'label': mne_raw.ch_names[i], 
                           'dimension': 'uV', 
                           'sample_frequency': sfreq, 
                           # 'physical_min': channels.min(), 
                           'physical_min': float(phys_min), 
                           # 'physical_max': channels.max(), 
                           'physical_max': float(phys_max), 
                           'digital_min':  dmin, 
                           'digital_max':  dmax, 
                           'transducer': '', 
                           'prefilter': ''}
        
            channel_info.append(ch_dict)
            
        # Utiliser l'API publique de MNE et des valeurs par défaut
        subj = (mne_raw.info.get('subject_info') or {})
        
        code = str(subj.get('id', '0'))
        # essaie différents champs possibles puis fallback
        name = (
            subj.get('name')
            or (" ".join([subj.get('first_name',''), subj.get('last_name','')]).strip())
            or subj.get('his_id')
            or 'noname'
        )
        f.setPatientCode(code)
        f.setPatientName(name)
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)
        
    except Exception as e:
        raise e
    finally:
        f.close()    
    return True
                   
#%% load data
folder_path = Path('C:/Users\yvan.nedelec\OneDrive - ICM\Documents\RE_yvan\projects\michelle_rescue/brainvision')

# channel to exclud bc they are off in the spectrum
excluded_ch = ['EOG_L', 'EOG_R', 'EMG']
# I removed manually tenEOG_R, EOG_R, EMG form the pick
picked_ch = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Pz', 'IO', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Oz']

fn = os.path.join(folder_path, "STS_Sleep06.vhdr")
raw = mne.io.read_raw_brainvision(fn, preload = True)
raw_psd = raw.compute_psd()
raw_data = raw.get_data(picks = picked_ch)*1e6 # to inspect data in µV
raw_std = np.std(raw_data, axis = 1)

#%% try to filt the data to see if it looks better
raw_filt = raw.copy().filter(l_freq = 0.5, h_freq = None)

raw_filt_lhp = raw.copy().filter(l_freq = 0.5, h_freq = 100)

raw_filt_lhp.notch_filter(freqs = np.arange(50, 250, 50))

#%% crop begining of raw to avoid the impedance check and the plugging of the channels (and the end because the recording looks weird by the end)
crop_raw = raw.copy().crop(tmin=38, tmax= 7310)
crop_raw_data = crop_raw.get_data(picks = picked_ch)*1e6 # to inspect data in µV
crop_raw_std = np.std(crop_raw_data, axis = 1)
crop_raw_psd = crop_raw.compute_psd()

# channels with std>7000
bad_idxs = [10,14,15,16,18,23,37,62] 
#%% plot timeseries raw
raw.plot(picks = picked_ch, n_channels = len(picked_ch))
# ==> even when we remove EOG and EMG (which have a bad psd compare to other channels), there is very high amplitude that is dispaly over the other channels, but that disapears when one channel is click (to mark as bad) and do not reappear if the channel is unmarked, but will reappear if we move across time on the recording

# bad channel manually identified
bads = [np.str_('P7'), np.str_('P4'), np.str_('P6'), np.str_('Oz'), np.str_('CP1'), np.str_('Fz'), np.str_('PO4'), np.str_('P2'), np.str_('CP3'), np.str_('CP4'), np.str_('EMG'), np.str_('F5'), np.str_('Fp1'), np.str_('Fpz'), np.str_('PO7'), np.str_('AF3'), np.str_('FC1'), np.str_('F1')]

#%% plot psd
raw_psd.plot(exclude = excluded_ch, scalings = dict(eeg=1000e-6))

crop_raw_psd.plot(exclude = excluded_ch)
# was hoping to clearly see outlier channels but non is really popping

#%% remove bad channels and crop on a clean portion of the signal 
raw_clean = raw_filt.copy().drop_channels(raw_filt.info['bads'])

raw_clean.crop(tmin = 1500, tmax = 1800)
# ==> it finally worked !!!! the data looks clear

#%% selecting a portion of the data that exceed the dynamic range
raw_exceed = raw_filt.copy().drop_channels(raw_filt.info['bads'])
raw_exceed.crop(tmin=54, tmax=62)
# the data still looks nice ^^' I have no idea why it didn't work before

#%% selecting data with only "good" channels 'quickly identified manually)

raw_badsout = raw_filt.copy().drop_channels(raw_filt.info['bads'])
# seems to work fine (although there is some clipping from time to time)
raw_badsout.crop(tmin = 60, tmax = 7310)
sfreq = raw_badsout.info['sfreq']
n_samples = raw_badsout.n_times
n_samples_to_keep = int(np.floor(n_samples / sfreq) * sfreq)
raw_badsout.crop(tmin=0, tmax=(n_samples_to_keep - 1) / sfreq)

#%% export to .edf
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_drauto.edf"), raw, fmt='edf', physical_range='auto', add_ch_type=False, overwrite=True, verbose=None)

mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_dr1000V.edf"), raw, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) # according to Chatty, +/-500 is interpreted in Volt by mne (explaining why we get flat channels, but it should not be the case for BrainAnalyzer)

# mne.export.export_raw(folder_path + '\STS_Sleep06_dr1000_test_inv.edf', raw, fmt='edf', physical_range=(500, -500), add_ch_type=False, overwrite=True, verbose=None)

# mne.export.export_raw(f"{folder_path}\STS_Sleep06_dr65536.edf", raw, fmt='edf', physical_range=(-32768, 32768), add_ch_type=False, overwrite=True, verbose=None)

# according to Chatty, le physical_range d'export_raw() doit être en Volt (car mne convertie toutes ses données en Volt), mais parès inspection des valeurs j'ai pas l'impression que ce soit le cas
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_dr1000uV.edf"), raw, fmt='edf', physical_range=(-500e-6, 500e-6), add_ch_type=False, overwrite=True, verbose=None) # according to Chatty, +/-500 is interpreted in Volt by mne (explaining why we get flat channels, but it should not be the case for BrainAnalyzer)

# trying a custom function to save mne raw object as edf (with manual cahnge to fix the dynamic range to 1000 uV)
write_mne_edf(raw, os.path.join(folder_path, "STS_Sleep06_customedfsave_dr1000uV.edf"), overwrite=True)
# => after inspection, there is the same problem as with the export_raw function.... (when plotting raw data)

# test to convert to edf after dropping bad channels and selecting only a clean portion of the signal
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_clean.edf"), raw_clean, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) 

# test with data that is exceeding the dynamic range or not
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_exceedNOT.edf"), raw_exceed, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) 
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_exceedYES.edf"), raw_exceed, fmt='edf', physical_range=(-250, 250), add_ch_type=False, overwrite=True, verbose=None) 

# test with data without bads
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_badsout.edf"), raw_badsout, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) 

# test with data witoubads and cropping the crazy beginning and ends
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_badsout_crop.edf"), raw_badsout, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) 


#%% cropping data to see if the pb comes from very high amplitude
cropped_raw = raw.copy().crop(tmin = 37, tmax = 47)
mne.export.export_raw(os.path.join(folder_path, "STS_Sleep06_dr1000V_cropped.edf"), cropped_raw, fmt='edf', physical_range=(-500, 500), add_ch_type=False, overwrite=True, verbose=None) # according to Chatty, +/-500 is interpreted in Volt by mne (explaining why we get flat channels, but it should not be the case for BrainAnalyzer)
# ==> seems like there is the same pb (but when we plot with mne python raw.plot() we see very high amplitude channels, that disapear if we mark one channel as bad and does not come back if we unmark it...)
#%% reload edf
raw_edf_auto = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_drauto.edf"), preload = True)
auto_psd = raw_edf_auto.compute_psd()
auto_data = raw_edf_auto.get_data()

raw_edf_dr1000V = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_dr1000V.edf"), preload = True)
dr1000V_psd = raw_edf_dr1000V.compute_psd()
dr1000V_data = raw_edf_dr1000V.get_data()

raw_edf_dr1000uV = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_dr1000uV.edf"), preload = True)
dr1000uV_psd = raw_edf_dr1000uV.compute_psd()
dr1000uV_data = raw_edf_dr1000uV.get_data()

raw_edf_manuallysaved = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_customedfsave_dr1000uV.edf"), preload = True)

raw_edf_cropped = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_dr1000V_cropped.edf"), preload = True)

raw_edf_clean = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_clean.edf"), preload = True)

raw_edf_exceedNOT = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_exceedNOT.edf"), preload = True)
raw_edf_exceedYES = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_exceedYES.edf"), preload = True)

raw_edf_badsout = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_badsout.edf"), preload = True)

raw_badsout = mne.io.read_raw_edf(os.path.join(folder_path, "STS_Sleep06_badsout_crop.edf"), preload = True)
# raw_edf_dr65536 = mne.io.read_raw_edf(f"{folder_path}\STS_Sleep06_dr65536.edf", preload = True)
# dr65536_psd = raw_edf_dr65536.compute_psd()
# dr65536_data = raw_edf_dr65536.get_data()
#%% check header
edf_to_inspect = os.path.join(folder_path, "STS_Sleep06_exceedYES.edf") # change the last part of the path to change file

edf_header = read_edf_header_custom(edf_to_inspect)
df_header = pd.DataFrame(edf_header) # convert to pd.dataframe for easy manipulation
# theoretical resolution (edf are 16bit files so the eeg signal can take 2^16 values within the dynamic range)
df_header['res_theoretical'] = (abs(pd.to_numeric(df_header['physical_min']))+abs(pd.to_numeric(df_header['physical_max'])))/pow(2,16)
# turn theoretical resolution to uV if dimension is mV (if no dimension, it is a mess)
df_header.loc[df_header['dimension'].str.contains('mv', case=False, na=False), 'res_theoretical'] *= 1000

#%% plot raw and psd
raw_inspect = raw_edf_cropped # select the data you want to plot
psd_inspect = dr1000uV_psd

raw_inspect.plot(picks = picked_ch, n_channels = len(picked_ch))
psd_inspect.plot(exclude = excluded_ch)

#%% plot one channel based on the index (identified from e.g. large std)

idx = 7

raw.plot(picks = raw.ch_names[idx])

#%% compare clean raw from brainvision vs edf
raw_clean.plot(scalings = dict(eeg=500e-6))
raw_edf_clean.plot(scalings = dict(eeg=500e-6))
# there is a weird last aprt that was added and automatically annoted bad

#%% compare the data
test = raw_clean.copy().get_data() - raw_edf_clean.copy().crop(tmax = raw_edf_clean.times[150000]).get_data()
