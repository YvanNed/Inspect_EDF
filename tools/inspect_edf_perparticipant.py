# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:33:14 2025

@author: yvan.nedelec
"""

###############################################################################
################## Extract .edf parameters per participants ###################
###############################################################################

# this script is following the same logic as the jupyter notebook inspect_edf_v1 
# but the output is a single file returning the parameters of each participants

#%% define thresholds for clipping and resoluton detection
dr_thres = 500 # in µV because data are converted to µV (even for EOG and MEG)
r_thres = 0.1 # in µV

#%% Import package and define custom functions
try:
    import os
    import re
    import html
    import chardet
    import warnings
    import numpy as np
    import pandas as pd
    import tkinter as tk
    from pathlib import Path
    from tkinter import filedialog

except ImportError as e:
    print("⚠️ Error: ", e)
else:
    print("✅ Packages and functions successfully imported!")

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

# custum function to get the path of a folder by selecting it from a browsing window 
def get_folder_path():
    root = tk.Tk()                      # initialyse graphic system to open a window
    root.withdraw()                     # hide the graphic initialysation
    root.attributes('-topmost', True)   # display the window in forefront
    root.update()                       # force to add the display parameter to the window
    folder = filedialog.askdirectory(title="Select the study folder that contains your dataset", parent = root)
    root.destroy()          # close the graphic initialysation
    return folder

# function to extract filter information from the string in headers
def extract_filter_value(s, tag):
    if pd.isna(s):
        return None
    match = re.search(rf'{tag}[:\s]*([\d\.]+)\s*', s, re.IGNORECASE)
    return float(match.group(1)) if match else None

# function to plot df columns more properly
def join_uniq(series):
    # Convertit tout en str (None -> "None", NaN -> "nan")
    vals = [str(x) for x in pd.unique(series)]
    return ", ".join(map(html.escape, vals))

# function to plot df columns more properly (not taking only unique values)
def join_values(series):
    # Convertit tout en str (None -> "None", NaN -> "nan")
    vals = [str(x) for x in series]
    return ", ".join(map(html.escape, vals))

#%% get data folder path and file list
folder_path = get_folder_path()
print(f'Selected folder: {folder_path}')

# list subfolder (list the content of the folder and check if it is a folder)
subfolder_list = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
print(f'Subfolders of the study: {subfolder_list}')
print('\nWe will now open and read informations from the edf files of your dataset')

# get the edf file list 
edf_files = [
    f for f in Path(folder_path).rglob('*.edf')
    if not f.name.startswith('._')
    ]

if not edf_files:
    print("⚠️ There is no .edf file in your folder")
else:
    print(f"\nThere is {len(edf_files)} .edf files in your folder!")
    
summary_path = f'{folder_path}/summary'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
    
#%% check if there is groups and session in the database
# check if there is a participants.tsv file to get different groups or sessions
# if there is not a participants.tsv we will try to infer groups from subfolder organization or filename components (additional part from subject number)
# in ICEBERG, subfolders define groups within the data folder
# in APOMORPHEE, suffixes define nights ("session") 
table_found = False
for root, dirs, files in os.walk(folder_path):
    if 'participants.tsv' in files:
        table_found = True        
        print(f"Table containing participants information found at: {os.path.join(root, 'participants.tsv')} ")
        subj_table_path = os.path.join(root, 'participants.tsv')

found_group = False
if table_found:
    subj_table = pd.read_csv(subj_table_path, sep = '\t', dtype={'participant_id': str, 'group': str})
    if "group" in subj_table.columns:
        found_group = True
        print("We will extract participant's group from it")
    else:
        print("No column 'group' was found in the table")
        print("Group will be inferred from subfolder organization or subfolder component") 
    
else:
    print("No table containing participants information (labelled 'participants.tsv') was found")
    print("If you have a table, please rename it 'participants.tsv' (and make sure you have columns labelled 'participant_id' and 'group' if any)")
    print("In the meantime, we will infer participant's group from subfolder organization or filename component in the next cells")
    subj_table = pd.DataFrame()

# Initialize a list of dataframes to store file info, which will be concatenated at the end (this is better for performance)
df_list = []
# Initialize an empty list for files that could not be read
failed_list = []

#%% loop over edf_file to extract information
with open(f"{summary_path}/EDF_perParticipant_report.html", "w", encoding="utf-8") as f:
    
    print("""<!doctype html>
        <html lang="fr">
        <head>
          <meta charset="utf-8">
          <title>List of parameters from EDF database per participant</title>
          <style>
            .indent1 { margin-left: 2ch; }  /* ~2 caractères */
            .indent2 { margin-left: 4ch; }  /* ~4 caractères */
            .indent3 { margin-left: 6ch; }  /* ~6 caractères */
          </style>
        </head>
        <body>""", file=f)
        
    print('<h1>List of parameters from EDF database per participant</h1>', file=f)
    print(f"<h2>selected database: {Path(folder_path).stem} </h2>", file=f)

    for e, edf_path in enumerate(edf_files):
        # print(f'file {e+1}/{len(edf_files)}, currently opening file: {edf_path}\n')
        
        # read file with the custom function
        try:
            edf_header = read_edf_header_custom(edf_path) 
            
            # get subject name (corresponding to file_name)
            sub_name = edf_path.stem
            
            # get subject group (from the parent folder because in the ICEBERG database subfolders were created per patient group)
            sub_folder = edf_path.parent.name # get the parent folder of the subject file (path)
            
            # create df from signal info
            df = pd.DataFrame(edf_header)
                
            # theoretical resolution (edf are 16bit files so the eeg signal can take 2^16 values within the dynamic range)
            df['res_theoretical'] = (abs(pd.to_numeric(df['physical_min']))+abs(pd.to_numeric(df['physical_max'])))/pow(2,16)
            # turn theoretical resolution to uV if dimension is mV (if no dimension, it is a mess)
            df.loc[df['dimension'].str.contains('mv', case=False, na=False), 'res_theoretical'] *= 1000
            
            # get filtering info in different columns
            df['lowpass']   = df['prefiltering'].apply(lambda x: extract_filter_value(x, 'LP'))
            df['highpass']  = df['prefiltering'].apply(lambda x: extract_filter_value(x, 'HP'))
            df['notch']  = df['prefiltering'].apply(lambda x: extract_filter_value(x, 'NOTCH'))
            
            # add subject info in the dataframe
            df['subject'] = sub_name
            df['sub_folder'] = sub_folder
            df['group'] = np.nan # initialyze column 'group' with NaN
            # get group from participants table if any (else group will be inferred from subfolder or filename extension later)
            if found_group:
                df['group'] = subj_table.loc[subj_table['participant_id'] == sub_name, 'group'].iloc[0]
    
            # extract filename component before and after subject number (so we assume subject name contains at least incrementing numbers that are at the beginning of the file name)  
            #   ^       → start of string  
            # (.*?)     → group 1: as few chars as possible, up to the first digit  
            # (\d+)     → group 2: the number itself  
            # (.*)      → group 3: the rest of the string  
            # $         → end of string
            pre_comp = sub_num = post_comp = np.nan
            pattern = re.compile(r'^(.*?)(\d+)(.*)$')
            m = pattern.match(sub_name)
            if m:
                pre_comp = m.group(1) or np.nan
                sub_num = m.group(2) or np.nan
                post_comp = m.group(3) or np.nan
            df['pre_fn_comp'] = pre_comp
            df['post_fn_comp'] = post_comp
            df['sub_num'] = sub_num
            
            df['path'] = str(edf_path)
            df['session'] = np.nan # session will be inferred later from file name component
            
            # select only the columns of interest
            df = df[['subject', 'group', 'session', 'path', 'sub_folder', 'sub_num', 'pre_fn_comp', 'post_fn_comp', 'channel', 'transducer_type', 'dimension', 'sampling_frequency', 
                 'highpass', 'lowpass', 'notch', 'physical_min', 'physical_max', 'res_theoretical']]
            ###################################################################
            # start printing information on the html file
            print(f'<h3>participant: {html.escape(df["subject"].dropna().astype(str).iloc[0])}</h3>', file=f)
            
            #######################################################################
            # extract EEG info
            # define common EEG label from the 10-10 convention
            COMMON_EEG_label = r'\bFp1\b|\bFpz\b|\bFp2\b|\bAF7\b|\bAF3\b|\bAFz\b|\bAF4\b|\bAF8\b|\bF7\b|\bF5\b|\bF3\b|\bF1\b|\bFz\b|\bF2\b|\bF4\b|\bF6\b|\bF8\b|\bFT7\b|\bFC5\b|\bFC3\b|\bFC1\b|\bFCz\b|\bFC2\b|\bFC4\b|\bFC6\b|\bFT8\b|\bT7\b|\bC5\b|\bC3\b|\bC1\b|\bCz\b|\bC2\b|\bC4\b|\bC6\b|\bT8\b|\bTP7\b|\bCP5\b|\bCP3\b|\bCP1\b|\bCPz\b|\bCP2\b|\bCP4\b|\bCP6\b|\bTP8\b|\bP7\b|\bP5\b|\bP3\b|\bP1\b|\bPz\b|\bP2\b|\bP4\b|\bP6\b|\bP8\b|\bPO7\b|\bPO5\b|\bPO3\b|\bPOz\b|\bPO4\b|\bPO6\b|\bPO8\b|\bO1\b|\bOz\b|\bO2\b|\bM1\b|\bM2\b|EEG'
            # select only EEG channels and return a warning if the number of participant is smaller/higher
            mask_ch = df['transducer_type'].str.contains(r'EEG|AGAGCL ELECTRODE', case = False, na=False) | df['channel'].str.contains(COMMON_EEG_label, case = False, na=False) # create a mask that returns true for lines containing either EEG/AGAGCL ELECTRODE in the transducer_type column or containing a common EEG label in the channel column
            df_ch = df[mask_ch]
            # remove the emg channels that were captured with the AGAGCL ELECTRODE transducer type 
            df_ch = df_ch[~df_ch['channel'].str.contains(r'emg|ecg|eog', case=False, na=False)] # the ~ allows to not select the selection (like ! in matlab)
            print('<h4 class="indent1">EEG:</h4>', file=f)
            if not df_ch.empty:
                print('<div class="indent2">', file=f)
                print(f'<p>EEG channels list: {join_values(df_ch["channel"])}</p>', file=f)
                print(f'<p>EEG sampling frequency: {join_uniq(df_ch["sampling_frequency"])}</p>', file=f)
                print(f'<p>EEG Filters (Hz): hp = {join_uniq(df_ch["highpass"])}; lp = {join_uniq(df_ch["lowpass"])}; notch = {join_uniq(df_ch["notch"])}</p>', file=f)
                print(f'<p>EEG unit: {join_uniq(df_ch["dimension"])}</p>', file=f)
                # inversion
                df_inv = df_ch[df_ch['physical_min'] > df_ch['physical_max']]
                if not df_inv.empty:
                    print('<p>EEG polarity: ⚠️ channels with inverted polarity detected!</p>', file=f)
                    print(f'<p class="indent3">channels: {join_uniq(df_inv["channel"])}</p>', file=f)
                else:
                    print('<p>EEG polarity: ✅ no inverted polarity detected in EEG channels!</p>', file=f)
                # clipping
                dr_mask = df_ch['res_theoretical']*pow(2,16) <= dr_thres
                bad_dr = df_ch[dr_mask]
                if not bad_dr.empty:
                    print(f'<p>EEG clipping: ⚠️ channels with clipping (dynamic range <= {dr_thres} µV) detected!</p>', file=f)
                    print(f'<p class="indent3">channels: {join_uniq(bad_dr["channel"])}</p>', file=f)
                else:
                    print(f'<p>EEG clipping: ✅ no clipping detected in EEG channels (dynamic range <= {dr_thres} µV)!</p>', file=f)
                # resolution
                r_mask = df_ch['res_theoretical'] >= r_thres
                bad_res = df_ch[r_mask]
                if not bad_res.empty:
                    print(f'<p>EEG resolution: ⚠️ channels with low resolution (>= {r_thres} µV) detected!</p>', file=f)
                    print(f'<p class="indent3">channels: {join_uniq(bad_res["channel"])}</p>', file=f)
                else:
                    print(f'<p>EEG resolution: ✅ no low resolution detected in EEG channels (>= {r_thres} µV)!</p>', file=f)
                print('</div>', file=f)
                # print('<p class="indent1">✅ Extraction of EEG parameters completed!</p>', file=f)
            else:
                print('<p class="indent2">❌ No EEG found </p>', file=f)
            
            #######################################################################
            
            #######################################################################
            # Extract EOG info
            mask_eog = df['channel'].str.contains(r'EOG', case = False, na=False) # create a mask that returns true for lines containing either EOG in the channel column
            df_eog = df[mask_eog]
            print('<h4 class="indent1">EOG:</h4>', file=f)
            if not df_eog.empty:
                 print('<div class="indent2">', file=f)
                 print(f'<p>EOG channels list: {join_values(df_eog["channel"])}</p>', file=f)
                 print(f'<p>EOG sampling frequency: {join_uniq(df_eog["sampling_frequency"])}</p>', file=f)
                 print(f'<p>EOG Filters (Hz): hp = {join_uniq(df_eog["highpass"])}; lp = {join_uniq(df_eog["lowpass"])}; notch = {join_uniq(df_eog["notch"])}</p>', file=f)
                 print(f'<p>EOG unit: {join_uniq(df_eog["dimension"])}</p>', file=f)
                 # inversion
                 df_inv_eog = df_eog[df_eog['physical_min'] > df_eog['physical_max']]
                 if not df_inv_eog.empty:
                     print('<p>EOG polarity: ⚠️ channels with inverted polarity detected!</p>', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(df_inv_eog["channel"])}</p>', file=f)
                 else:
                     print('<p>EOG polarity: ✅ no inverted polarity detected in EOG channels!</p>', file=f)
                 # clipping
                 dr_mask_eog = df_eog['res_theoretical']*pow(2,16) <= dr_thres
                 bad_dr_eog = df_eog[dr_mask_eog]
                 if not bad_dr_eog.empty:
                     print(f'<p>EOG clipping: ⚠️ channels with clipping (dynamic range <= {dr_thres} µV) detected!</p>', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(bad_dr_eog["channel"])}</p>', file=f)
                 else:
                     print(f'<p>EOG clipping: ✅ no clipping detected in EOG channels (dynamic range <= {dr_thres} µV)!</p>', file=f)
                 # resolution
                 r_mask_eog = df_eog['res_theoretical'] >= r_thres
                 bad_res_eog = df_eog[r_mask_eog]
                 if not bad_res_eog.empty:
                     print(f'<p>EOG resolution: ⚠️ channels with low resolution (>= {r_thres} µV) detected!</p>', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(bad_res_eog["channel"])}</p>', file=f)
                 else:
                     print(f'<p>EOG resolution: ✅ no low resolution detected in EOG channels (>= {r_thres} µV)!</p>', file=f)
                 print('</div>', file=f) 
                 # print(f'<p class="indent1">✅ Extraction of EOG parameters from participant {df_eog["subject"].unique()} completed!</p>', file=f)
            else:
                 print('<p class="indent2">❌ No EOG found <p>', file=f)   
            
            #######################################################################
            
            #######################################################################
            # Extract ECG info
            mask_ecg = df['channel'].str.contains(r'ecg', case = False, na=False) # create a mask that returns true for lines containing either ecg in the channel column
            df_ecg = df[mask_ecg]

            print('<h4 class="indent1">ECG:</h4>', file=f)
            if not df_ecg.empty:
                 print('<div class="indent2">', file=f)
                 print(f'<p>ECG channels list: {join_values(df_ecg["channel"])}</p>', file=f)
                 print(f'<p>ECG sampling frequency: {join_uniq(df_ecg["sampling_frequency"])}</p>', file=f)
                 print(f'<p>ECG Filters (Hz): hp = {join_uniq(df_ecg["highpass"])}; lp = {join_uniq(df_ecg["lowpass"])}; notch = {join_uniq(df_ecg["notch"])}</p>', file=f)
                 print(f'<p>ECG unit: {join_uniq(df_ecg["dimension"])}', file=f)
                 # inversion
                 df_inv_ecg = df_ecg[df_ecg['physical_min'] > df_ecg['physical_max']]
                 if not df_inv_ecg.empty:
                     print('<p>ECG polarity: ⚠️ channels with inverted polarity detected!', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(df_inv_ecg["channel"])}</p>', file=f)
                 else:
                     print('<p>ECG polarity: ✅ no inverted polarity detected in ECG channels!', file=f)
                 # clipping
                 dr_mask_ecg = df_ecg['res_theoretical']*pow(2,16) <= dr_thres
                 bad_dr_ecg = df_ecg[dr_mask_ecg]
                 if not bad_dr_ecg.empty:
                     print(f'<p>ECG clipping: ⚠️ channels with clipping (dynamic range <= {dr_thres} µV) detected!', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(bad_dr_ecg["channel"])}</p>', file=f)
                 else:
                     print(f'<p>ECG clipping: ✅ no clipping detected in ECG channels (dynamic range <= {dr_thres} µV)!', file=f)
                 # resolution
                 r_mask_ecg = df_ecg['res_theoretical'] >= r_thres
                 bad_res_ecg = df_ecg[r_mask_ecg]
                 if not bad_res_ecg.empty:
                     print(f'<p>ECG resolution: ⚠️ channels with low resolution (>= {r_thres} µV) detected!', file=f)
                     print(f'<p class="indent3">channels: {join_uniq(bad_res_ecg["channel"])}</p>', file=f)
                 else:
                     print(f'<p>ECG resolution: ✅ no low resolution detected in ECG channels (>= {r_thres} µV)!', file=f)
                 print('</div>', file=f)  
                 # print(f'<p class="indent1">✅ Extraction of ECG parameters from participant {df_ecg["subject"].unique()} completed!</p>', file=f)
            else:
                 print('<p class="indent2">❌ No ECG found </p>', file=f) 
            
            #######################################################################
            
            # store subject data
            df_list.append(df)
    
        except UnicodeDecodeError as e:
            print(f"<p class='indent1'>⚠️ Encoding problem for {edf_path}</p>", file=f)
            failed_list.append((edf_path, 'encoding'))
        except Exception as e:
            # tb = traceback.format_exc()
            print(f"<p class='indent1'>❌ Unexpected problem for {edf_path} : {e}</p>", file=f)
            failed_list.append((edf_path, 'other'))
    
    print("</body></html>", file=f)
   
# concatenate dataframe into one and only
with warnings.catch_warnings(): # this is to skip a warning not affecting our operation
    warnings.simplefilter("ignore", FutureWarning)
    # df_full = pd.concat(df_list, ignore_index=True)

# save the failed list if not empty:
failed_df = pd.DataFrame(failed_list)
if not failed_df.empty:
    failed_df.to_csv(f'{summary_path}/failed_edf_read.tsv', sep = '\t')
    print(f'\nSaving the list of files that could not be read to: \n{summary_path}/failed_edf_read.tsv')    

















