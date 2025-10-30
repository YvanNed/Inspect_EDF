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

print("\n----------------------------------------------------------------------")
print("------------------- Inspection of an EDF database --------------------")
print("----------------------------------------------------------------------")
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
    print("\nPackages and functions successfully imported!")

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
print(f'\nSelected folder: {folder_path}')

# list subfolder (list the content of the folder and check if it is a folder)
subfolder_list = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
# print(f'Subfolders of the study: {subfolder_list}')
print('We will now open and read informations from the edf files of your dataset')

# get the edf file list 
edf_files = [
    f for f in Path(folder_path).rglob('*.edf')
    if not f.name.startswith('._')
    ]

if not edf_files:
    print("⚠️ There is no .edf file in your folder")
else:
    print(f"\nThere is {len(edf_files)} .edf files in your folder.\n")
    
summary_path = f'{folder_path}/summary'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
    
#%% check if there is groups and session in the database

# THIS IS CURRENTLY REMOVED FROM THE SCRIPT (only the printing lines)
# because it does not take into account database organisation with one folder per subject

# check if there is a participants.tsv file to get different groups or sessions
# if there is not a participants.tsv we will try to infer groups from subfolder organization or filename components (additional part from subject number)
# in ICEBERG, subfolders define groups within the data folder
# in APOMORPHEE, suffixes define nights ("session") 
table_found = False
for root, dirs, files in os.walk(folder_path):
    if 'participants.tsv' in files:
        table_found = True        
        # print(f"Table containing participants information found at: {os.path.join(root, 'participants.tsv')} ")
        subj_table_path = os.path.join(root, 'participants.tsv')

found_group = False
if table_found:
    subj_table = pd.read_csv(subj_table_path, sep = '\t', dtype={'participant_id': str, 'group': str})
    if "group" in subj_table.columns:
        found_group = True
        # print("We will extract participant's group from it")
    else:
        print('')
        # print("No column 'group' was found in the table")
        # print("Group will be inferred from subfolder organization or subfolder component") 
    
else:
    subj_table = pd.DataFrame()
    # print("No table containing participants information (labelled 'participants.tsv') was found")
    # print("If you have a table, please rename it 'participants.tsv' (and make sure you have columns labelled 'participant_id' and 'group' if any)")
    # print("In the meantime, we will infer participant's group from subfolder organization or filename component.\n")

# Initialize a list of dataframes to store file info, which will be concatenated at the end (this is better for performance)
df_list = []
# Initialize an empty list for files that could not be read
failed_list = []

#%% loop over edf_file to extract information

# to debug on one subject
# e = 0
# edf_path = edf_files[e]

for e, edf_path in enumerate(edf_files):
    print(f'file {e+1}/{len(edf_files)}, currently opening file: {edf_path.name}')
    
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
        df = df[['subject', 'group', 'session', 'path', 'sub_folder', 'sub_num', 'pre_fn_comp', 'post_fn_comp', 'channel', 'transducer_type', 
                 'dimension', 'sampling_frequency', 'prefiltering', 'highpass', 'lowpass', 'notch', 'physical_min', 'physical_max', 'res_theoretical']]
        
        # store subject data
        df_list.append(df)

    except UnicodeDecodeError as e:
        print(f"❌ Encoding problem for {edf_path}")
        failed_list.append((edf_path, 'encoding'))
    except Exception as e:
        # tb = traceback.format_exc()
        print(f"❌ Unexpected problem for {edf_path} : {e}")
        failed_list.append((edf_path, 'other'))
    
   
# concatenate dataframe into one and only
with warnings.catch_warnings(): # this is to skip a warning not affecting our operation
    warnings.simplefilter("ignore", FutureWarning)
    df_full = pd.concat(df_list, ignore_index=True)

# save the failed list if not empty:
failed_df = pd.DataFrame(failed_list)
if not failed_df.empty:
    failed_df.to_csv(f'{summary_path}/failed_edf_read.tsv', sep = '\t')
    print(f'\nSaving the list of files that could not be read to: \n{summary_path}/failed_edf_read.tsv')    
    
    
#%% get group informaton before saving the summary_table

# Commented for now because not handling all possible situation (1 folder per subject, detecting different groups if not starting by participant number)
# get group information, from participants.tsv file, sub_folder, or filename component

# print("Get group information:")
# if df_full['group'].isna().all():
#     print("The column 'group' is empty, we will infer group from subfolder, if any...")
#     if len(df_full['sub_folder'].unique()) > 1:
#         df_full['group'] = df_full['sub_folder']
#         print(">>> Group inferred from folders within the database <<<")
#     else:
#         print("There is no distinct folders for groups.")
#         print("Trying to infer group from filename component...")
#         # looping across subject number (and not subject filename) to test if there are multiple filename components per subject (to disentangle groups from session)  
#         count_precomp = np.zeros(len(df_full['sub_num'].unique()))
#         count_postcomp = np.zeros(len(df_full['sub_num'].unique()))
#         for sn, sub_num in enumerate(df_full['sub_num'].unique()):
#             df_sub = df_full[df_full['sub_num'] == sub_num]
#             count_precomp[sn] = len(df_sub['pre_fn_comp'].unique())
#             count_postcomp[sn] = len(df_sub['post_fn_comp'].unique())
#         # fn component is a group if within subject there is only one component, but there are multiple components between subject
#         # 1st, try for component before the subject number, 2nd try for component after the subject number 
#         if len(df_full['pre_fn_comp'].unique()) > 1 and count_precomp.mean() == 1:
#             df_full['group'] = df_full['pre_fn_comp']
#             print(">>> Group inferred from filename component (before subject number) <<<")
#         elif len(df_full['post_fn_comp'].unique()) > 1 and count_postcomp.mean() == 1:
#             df_full['group'] = df_full['post_fn_comp']
#             print(">>> Group inferred from filename component (after subject number) <<<")
#         else:
#             print("Did not succeed to identify group from filename component.")
#             print("It seems that there is only one group in the study!")
# else:
#     print(">>> Group information coming from participants.tsv <<<")

# print("\nGet session information")
# if len(df_full['pre_fn_comp'].unique()) > 1 and count_precomp.mean() > 1:
#     print(">>> Session inferred from filename component (before subject number) <<<")
#     df_full['session'] = df_full['pre_fn_comp']
# elif len(df_full['post_fn_comp'].unique()) > 1 and count_postcomp.mean() > 1:
#     print(">>> Session inferred from filename component (after subject number) <<<")
#     df_full['session'] = df_full['post_fn_comp']
# else:
#     print("It seems that there is only one session in the study")

# save summary table containing full info
df_full.to_csv(f'{summary_path}/FULL_summary.tsv', sep = '\t')
print(f'\nSaving full informations from the dataset to:\n{summary_path}/FULL_summary.tsv')

# print("\n\nDataset information:")
# print(f"- Number of files: {len(df_full['subject'].unique())}")
# print(f"- Number of participants: {len(df_full['sub_num'].unique())}")
# print(f"- Number of groups: {len(df_full['group'].unique())}")
# print(f"- Number of sessions: {len(df_full['session'].unique())}")

# if len(df_full['group'].unique()) > 1:
#     print("\nParticipants per groups:")
#     print(df_full.drop_duplicates().groupby('group').agg(n_subjects=('subject', 'nunique')))
    
#%% open/create a html file to create the report
with open(f"{summary_path}/EDF_inspection_report.html", "w", encoding="utf-8") as f:
    
    print("""<!doctype html>
        <html lang="fr">
        <head>
          <meta charset="utf-8">
          <title>Fast inspection of .edf files from {Path(folder_path).stem}</title>
          <style>
            .indent1 { margin-left: 2ch; }  /* ~2 caractères */
            .indent2 { margin-left: 4ch; }  /* ~4 caractères */
            .indent3 { margin-left: 6ch; }  /* ~6 caractères */
          </style>
        </head>
        <body>""", file=f)
        
    print(f'<h1>Fast inspection of .edf files from {Path(folder_path).stem}</h1>', file=f)
    
    ###########################################################################
    # start printing global information of the database
    print('<h2>Global information:</h2>', file=f)
    print(f'<p class="indent1">- Number of files: {len(df_full["subject"].unique())}<p>', file=f)
    print(f'<p class="indent1">- Number of participants: {len(df_full["sub_num"].unique())}<p>', file=f)
    
    ###########################################################################
    # extract EEG info
    # define common EEG label from the 10-10 convention
    COMMON_EEG_label = r'\bFp1\b|\bFpz\b|\bFp2\b|\bAF7\b|\bAF3\b|\bAFz\b|\bAF4\b|\bAF8\b|\bF7\b|\bF5\b|\bF3\b|\bF1\b|\bFz\b|\bF2\b|\bF4\b|\bF6\b|\bF8\b|\bFT7\b|\bFC5\b|\bFC3\b|\bFC1\b|\bFCz\b|\bFC2\b|\bFC4\b|\bFC6\b|\bFT8\b|\bT7\b|\bC5\b|\bC3\b|\bC1\b|\bCz\b|\bC2\b|\bC4\b|\bC6\b|\bT8\b|\bTP7\b|\bCP5\b|\bCP3\b|\bCP1\b|\bCPz\b|\bCP2\b|\bCP4\b|\bCP6\b|\bTP8\b|\bP7\b|\bP5\b|\bP3\b|\bP1\b|\bPz\b|\bP2\b|\bP4\b|\bP6\b|\bP8\b|\bPO7\b|\bPO5\b|\bPO3\b|\bPOz\b|\bPO4\b|\bPO6\b|\bPO8\b|\bO1\b|\bOz\b|\bO2\b|\bM1\b|\bM2\b|EEG'
    # select only EEG channels and return a warning if the number of participant is smaller/higher
    mask_ch = df_full['transducer_type'].str.contains(r'EEG|AGAGCL ELECTRODE', case = False, na=False) | df_full['channel'].str.contains(COMMON_EEG_label, case = False, na=False) # create a mask that returns true for lines containing either EEG/AGAGCL ELECTRODE in the transducer_type column or containing a common EEG label in the channel column
    df_full_ch = df_full[mask_ch]
    # remove the emg channels that were captured with the AGAGCL ELECTRODE transducer type 
    df_full_ch = df_full_ch[~df_full_ch['channel'].str.contains(r'emg|ecg|eog', case=False, na=False)] # the ~ allows to not select the selection (like ! in matlab)
    
    # Check if the number of participants with only EEG is the same as df_full. 
    # If not, it might be because the transducer type was no correctly detected. 
    # One possibility is to add the type of transducer to the condition line 2 of this cell.
    if len(df_full['subject'].unique()) > len(df_full_ch['subject'].unique()):
        # identify missing subjects
        missing_sub = set(df_full['subject'].unique()) - set(df_full_ch['subject'].unique())
        print('\n!!! There is less participants in the dataset with only EEGs !!!')
        # print(f'Missing participants: {missing_sub}')
        print("Either these participants don't have EEGs.")
        print("Or the transducer type was not correctly detected.")
        # get df of missing sub to save and inspect
        df_miss = df_full[df_full['subject'].isin(missing_sub)]
        df_miss.to_csv(f'{summary_path}/EEG_missing_edf.tsv', sep = '\t')
        print(f'Saving informations from participants missing EEGs to:\n{summary_path}/EEG_missing_edf.tsv')
        print('Please inspect the file, and specifically the column transducer_type if they should have EEGs')
        
    elif len(df_full['subject'].unique()) < len(df_full_ch['subject'].unique()):
        print('\n!!! There is more participants in the dataset with only EEGs !!!')
        print('This should not be the case.')
        print('Please inspect what is happening in a code editor (spyder..), or ask Yvan.')
        more_sub = set(df_full_ch['subject'].unique()) - set(df_full['subject'].unique())
        df_more = df_full_ch[df_full_ch['subject'].isin(more_sub)]
        df_more.to_csv(f'{summary_path}/EEG_toomany_edf.tsv', sep = '\t')
        print(f'Saving informations from participants suspect EEGs to:\n{summary_path}/EEG_toomany_edf.tsv')
        
    # saving info from eeg
    df_full_ch.to_csv(f'{summary_path}/EEG_summary.tsv', sep = '\t')
    print(f'\nSaving informations from EEGs to:\n{summary_path}/EEG_summary.tsv')
    
    print('<h2"><b>EEG:</b></h2>', file=f)
    if not df_full_ch.empty:
        print('<div class="indent1">', file=f)
        
        # EEG configuration check______________________________________________
        ch_per_sub = df_full_ch.groupby('subject')['channel'].apply(lambda x: tuple(sorted(set(x))))

        # identify the channel configuration of each participant and store them in a dict to print per channel config
        ch_config_dict = {}
        for config in ch_per_sub.unique():
            sub = ch_per_sub[ch_per_sub == config].index.tolist()
            ch_config_dict[config] = sub
        
        if len(ch_config_dict) > 1:
            print(f'<p>⚠️ {len(ch_config_dict)} different EEG configurations found</p>', file=f)
            print('<p class="indent2">You will have to harmonize the number and the name of channels for your analysis</p>', file=f)
            print('<p class="indent2"><i>-WIP- You can use the notebook "TBA" to do it -WIP-</i></p>', file=f)
        else:
            print('<p>✅ All your participants have the same EEG channels</p>', file=f)
        #______________________________________________________________________
        
        # EEG sampling frequency check_________________________________________
        # the sampling frequency configuration
        sf_per_sub = df_full_ch.groupby('subject')['sampling_frequency'].apply(lambda x: tuple(sorted(set(x))))
        # identify the sampling frequency configuration of each participant and store them in a dict to print per sampling configuration config
        sf_config_dict = {}
        for config in sf_per_sub.unique():
            sub = sf_per_sub[sf_per_sub == config].index.tolist()
            sf_config_dict[config] = sub
        
        if len(sf_config_dict) > 1:
            print(f'<p>⚠️ {len(sf_config_dict)} different EEG sampling frequencies found: {join_uniq(df_full_ch["sampling_frequency"])} Hz</p>', file=f)
            print('<p class="indent2">You can either re-export your data or downsample to a common sampling frequency for your analysis</p>', file=f)
            print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported</i></p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EEG sampling frequency: {join_uniq(df_full_ch["sampling_frequency"])} Hz</p>', file=f)
        #______________________________________________________________________
        
        # EEG filters check____________________________________________________
        # Get the list of participants with different filtering parameters
        # 1st replace NaN because groupby does not like NaN
        df_filt = df_full_ch.copy()
        df_filt[['lowpass', 'highpass', 'notch']] = df_filt[['lowpass', 'highpass', 'notch']].fillna('missing')
        
        config_filters = (
            df_filt.groupby(['lowpass', 'highpass', 'notch'])['subject']
            .apply(lambda x: sorted(set(x)))
            .reset_index(name = 'subjects')
        )
        
        if len(config_filters) > 1:
            print(f'<p>⚠️ {len(config_filters)} different EEG filters configurations found:</p>', file=f)
            for idx, row in config_filters.iterrows():
                print(f'<p class="indent3">filter config. {idx+1}: hp = {row["highpass"]} Hz; lp = {row["lowpass"]} Hz; notch = {row["notch"]} Hz<br></p>', file=f)
            print('<p class="indent2">You can either re-export your data or filter your data to a common fequency</p>', file=f)
            print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported or filtered</i></p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EEG filters: hp = {join_uniq(config_filters["highpass"])} Hz; lp = {join_uniq(config_filters["lowpass"])} Hz; notch = {join_uniq(config_filters["notch"])} Hz</p>', file=f)
        #______________________________________________________________________
        
        # EEG units check______________________________________________________
        if len(df_full_ch['dimension'].unique()) > 1:
            print(f'<p>⚠️ {len(df_full_ch["dimension"].unique())} different EEG units found: {join_uniq(df_full_ch["dimension"])}</p>', file=f)
            print('<p class="indent2">Before analyzing, make sure that your software (MNE, FIELDTRIP) correctly read your data unit</p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EEG unit: {join_uniq(df_full_ch["dimension"])}</p>', file=f)
        #______________________________________________________________________
        
        # EEG inversion check__________________________________________________
        df_full_inv = df_full_ch[df_full_ch['physical_min'] > df_full_ch['physical_max']]
        if not df_full_inv.empty:
            print('<p><b>EEG polarity</b>: ❌ EGGs with inverted polarity detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(df_full_inv["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            df_full_inv.to_csv(f'{summary_path}/EEG_inverted_polarity.tsv', sep = '\t')
            print(f'\nSaving informations from inverted polarity EEGs to:\n{summary_path}/EEG_inverted_polarity.tsv')
        else:
            print('<p><b>EEG polarity</b>: ✅ no inverted polarity detected in EEGs!</p>', file=f)
        #______________________________________________________________________
        
        # EEG clipping check___________________________________________________
        dr_mask = df_full_ch['res_theoretical']*pow(2,16) <= dr_thres
        bad_dr = df_full_ch[dr_mask]
        if not bad_dr.empty:
            print(f'<p><b>EEG clipping</b>: ❌ EEGs with clipping (dynamic range <= {dr_thres} µV) detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(bad_dr["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            bad_dr.to_csv(f'{summary_path}/EEG_bad_dynamic_range.tsv', sep = '\t')
            print(f'\nSaving informations fram bad dynamic range EEGs to:\n{summary_path}/EEG_bad_dynamic_range.tsv')
        else:
            print(f'<p><b>EEG clipping</b>: ✅ no clipping detected in EEGs (dynamic range <= {dr_thres} µV)!</p>', file=f)
        #______________________________________________________________________
        
        # EEG resolution check_________________________________________________
        r_mask = df_full_ch['res_theoretical'] >= r_thres
        bad_res = df_full_ch[r_mask]
        if not bad_res.empty:
            print(f'<p><b>EEG resolution</b>: ❌ EEGs with low resolution (>= {r_thres} µV) detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(bad_res["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            bad_res.to_csv(f'{summary_path}/EEG_bad_resolution.tsv', sep = '\t')
            print(f'\nSaving informations from bad resolution EEGs to:\n{summary_path}/EEG_bad_resolution.tsv')
        else:
            print(f'<p><b>EEG resolution</b>: ✅ no low resolution detected in EEGs (>= {r_thres} µV)!</p>', file=f)
        #______________________________________________________________________
        print('</div>', file=f)
        # print('<p class="indent1">✅ Extraction of EEG parameters completed!</p>', file=f)
    else:
        print('<p class="indent1">❌ No EEG found </p>', file=f)
    
    ###########################################################################
    
    # ###########################################################################
    # Extract EOG info
    mask_eog = df_full['channel'].str.contains(r'EOG', case = False, na=False) # create a mask that returns true for lines containing either EOG in the channel column
    df_full_eog = df_full[mask_eog]
    
    # Check if the number of participants with only EOG is the same as df_full. 
    # If not, it might be because the transducer type was no correctly detected. 
    # One possibility is to add the type of transducer to the condition line 2 of this cell.
    if len(df_full['subject'].unique()) > len(df_full_eog['subject'].unique()):
        # identify missing subjects
        missing_sub = set(df_full['subject'].unique()) - set(df_full_eog['subject'].unique())
        print('\n!!! There is less participants in the dataset with only EOGs !!!')
        # print(f'Missing participants: {missing_sub}')
        print("Either these participants don't have EOGs.")
        print("Or the transducer type was not correctly detected.")
        # get df of missing sub to save and inspect
        df_full_eogmiss = df_full[df_full['subject'].isin(missing_sub)]
        df_full_eogmiss.to_csv(f'{summary_path}/EOG_missing_edf.tsv', sep = '\t')
        print(f'Saving informations from participants missing EOGs to:\n{summary_path}/EOG_missing_edf.tsv')
        print('Please inspect the file, and specifically the column transducer_type if they should have EOGs')
        
    elif len(df_full['subject'].unique()) < len(df_full_eog['subject'].unique()):
        print('\n!!! There is more participants in the dataset with only EOGs !!!')
        print('This should not be the case.')
        print('Please inspect what is happening in a code editor (spyder..), or ask Yvan.')
        more_sub = set(df_full_eog['subject'].unique()) - set(df_full['subject'].unique())
        df_more = df_full_eog[df_full_eog['subject'].isin(more_sub)]
        df_more.to_csv(f'{summary_path}/EOG_toomany_edf.csv', sep = '\t')
        print(f'Saving informations from participants suspect EOGs to:\n{summary_path}/EOG_toomany_edf.tsv')
    
    # saving info from EOG
    df_full_eog.to_csv(f'{summary_path}/EOG_summary.tsv', sep = '\t')
    print(f'\nSaving informations from EOGs to:\n{summary_path}/EOG_summary.tsv')
    
    print('<h2"><b>EOG:</b></h2>', file=f)
    if not df_full_eog.empty:
        print('<div class="indent1">', file=f)
        
        # EOG configuration check______________________________________________
        eog_per_sub = df_full_eog.groupby('subject')['channel'].apply(lambda x: tuple(sorted(set(x))))

        # identify the channel configuration of each participant and store them in a dict to print per channel config
        eog_config_dict = {}
        for config in eog_per_sub.unique():
            sub = eog_per_sub[eog_per_sub == config].index.tolist()
            eog_config_dict[config] = sub
        
        if len(eog_config_dict) > 1:
            print(f'<p>⚠️ {len(eog_config_dict)} different EOG configurations found</p>', file=f)
            print('<p class="indent2">You will have to harmonize the number and the name of channels for your analysis</p>', file=f)
            print('<p class="indent2"><i>-WIP- You can use the notebook "TBA" to do it -WIP-</i></p>', file=f)
        else:
            print('<p>✅ All your participants have the same EOG channels</p>', file=f)
        #______________________________________________________________________
        
        # EOG sampling frequency check_________________________________________
        # the sampling frequency configuration
        sfeog_per_sub = df_full_eog.groupby('subject')['sampling_frequency'].apply(lambda x: tuple(sorted(set(x))))
        # identify the sampling frequency configuration of each participant and store them in a dict to print per sampling configuration config
        sfeog_config_dict = {}
        for config in sfeog_per_sub.unique():
            sub = sfeog_per_sub[sfeog_per_sub == config].index.tolist()
            sfeog_config_dict[config] = sub
        
        if len(sfeog_config_dict) > 1:
            print(f'<p>⚠️ {len(sfeog_config_dict)} different EOG sampling frequencies found: {join_uniq(df_full_eog["sampling_frequency"])} Hz</p>', file=f)
            print('<p class="indent2">You can either re-export your data or downsample to a common sampling frequency for your analysis</p>', file=f)
            print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported</i></p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EOG sampling frequency: {join_uniq(df_full_eog["sampling_frequency"])} Hz</p>', file=f)
        #______________________________________________________________________
        
        # EOG filters check____________________________________________________
        # Get the list of participants with different filtering parameters
        # 1st replace NaN because groupby does not like NaN
        df_eogfilt = df_full_eog.copy()
        df_eogfilt[['lowpass', 'highpass', 'notch']] = df_eogfilt[['lowpass', 'highpass', 'notch']].fillna('missing')
        
        config_eogfilters = (
            df_eogfilt.groupby(['lowpass', 'highpass', 'notch'])['subject']
            .apply(lambda x: sorted(set(x)))
            .reset_index(name = 'subjects')
        )
        
        if len(config_eogfilters) > 1:
            print(f'<p>⚠️ {len(config_eogfilters)} different EOG filters configurations found:</p>', file=f)
            for idx, row in config_eogfilters.iterrows():
                print(f'<p class="indent3">filter config. {idx+1}: hp = {row["highpass"]} Hz; lp = {row["lowpass"]} Hz; notch = {row["notch"]} Hz<br></p>', file=f)
            print('<p class="indent2">You can either re-export your data or filter your data to a common fequency</p>', file=f)
            print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported or filtered</i></p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EOG filters: hp = {join_uniq(config_eogfilters["highpass"])} Hz; lp = {join_uniq(config_eogfilters["lowpass"])} Hz; notch = {join_uniq(config_eogfilters["notch"])} Hz</p>', file=f)
        #______________________________________________________________________
        
        # EOG units check______________________________________________________
        if len(df_full_eog['dimension'].unique()) > 1:
            print(f'<p>⚠️ {len(df_full_eog["dimension"].unique())} different EOG units found: {join_uniq(df_full_eog["dimension"])}</p>', file=f)
            print('<p class="indent2">Before analyzing, make sure that your software (MNE, FIELDTRIP) correctly read your data unit</p>', file=f)
        else:
            print(f'<p>✅ All your participants have the same EOG unit: {join_uniq(df_full_eog["dimension"])}</p>', file=f)
        #______________________________________________________________________
        
        # EOG inversion check__________________________________________________
        df_full_eoginv = df_full_eog[df_full_eog['physical_min'] > df_full_eog['physical_max']]
        if not df_full_eoginv.empty:
            print('<p><b>EOG polarity</b>: ❌ EOGs with inverted polarity detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(df_full_eoginv["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            df_full_eoginv.to_csv(f'{summary_path}/EOG_inverted_polarity.tsv', sep = '\t')
            print(f'\nSaving informations from inverted polarity EOGs to:\n{summary_path}/EOG_inverted_polarity.tsv')
        else:
            print('<p><b>EOG polarity</b>: ✅ no inverted polarity detected in EOGs!</p>', file=f)
        #______________________________________________________________________
        
        # EOG clipping check___________________________________________________
        dr_eogmask = df_full_eog['res_theoretical']*pow(2,16) <= dr_thres
        bad_eogdr = df_full_eog[dr_eogmask]
        if not bad_eogdr.empty:
            print(f'<p><b>EOG clipping</b>: ❌ EOGs with clipping (dynamic range <= {dr_thres} µV) detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(bad_eogdr["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            bad_eogdr.to_csv(f'{summary_path}/EOG_bad_dynamic_range.tsv', sep = '\t')
            print(f'\nSaving informations from bad dynamic range EOGs to:\n{summary_path}/EOG_bad_dynamic_range.tsv')
        else:
            print(f'<p><b>EOG clipping</b>: ✅ no clipping detected in EOGs (dynamic range <= {dr_thres} µV)!</p>', file=f)
        #______________________________________________________________________
        
        # EOG resolution check_________________________________________________
        r_eogmask = df_full_eog['res_theoretical'] >= r_thres
        bad_eogres = df_full_eog[r_eogmask]
        if not bad_eogres.empty:
            print(f'<p><b>EOG resolution</b>: ❌ EOGs with low resolution (>= {r_thres} µV) detected!</p>', file=f)
            print(f'<p class="indent2">It concerns files: {join_uniq(bad_eogres["subject"])}</p>', file=f)
            print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
            bad_eogres.to_csv(f'{summary_path}/EOG_bad_resolution.tsv', sep = '\t')
            print(f'\nSaving informations from bad resolution EOGs to:\n{summary_path}/EOG_bad_resolution.tsv')
        else:
            print(f'<p><b>EOG resolution</b>: ✅ no low resolution detected in EOGs (>= {r_thres} µV)!</p>', file=f)
        #______________________________________________________________________
        print('</div>', file=f)
        # print('<p class="indent1">✅ Extraction of EOG parameters completed!</p>', file=f)
    else:
        print('<p class="indent1">❌ No EOG found </p>', file=f)
    # ###########################################################################
    
    # ###########################################################################
    # # # Extract ECG info
    # # I am removing (commenting) ECG infromation exctraction for now because it makes the output too heavy
    # mask_ecg = df_full['channel'].str.contains(r'ecg', case = False, na=False) # create a mask that returns true for lines containing either ecg in the channel column
    # df_full_ecg = df_full[mask_ecg]

    # # Check if the number of participants with only ECG is the same as df_full. 
    # # If not, it might be because the transducer type was no correctly detected. 
    # # One possibility is to add the type of transducer to the condition line 2 of this cell.
    # if len(df_full['subject'].unique()) > len(df_full_ecg['subject'].unique()):
    #     # identify missing subjects
    #     missing_sub = set(df_full['subject'].unique()) - set(df_full_ecg['subject'].unique())
    #     print('\n!!! There is less participants in the dataset with only ECGs !!!')
    #     # print(f'Missing participants: {missing_sub}')
    #     print("Either these participants don't have ECGs.")
    #     print("Or the transducer type was not correctly detected.")
    #     # get df of missing sub to save and inspect
    #     df_full_ecgmiss = df_full[df_full['subject'].isin(missing_sub)]
    #     df_full_ecgmiss.to_csv(f'{summary_path}/ECG_missing_edf.tsv', sep = '\t')
    #     print(f'Saving informations from participants missing ECGs to:\n{summary_path}/ECG_missing_edf.tsv')
    #     print('Please inspect the file, and specifically the column transducer_type if they should have ECGs')
    # elif len(df_full['subject'].unique()) < len(df_full_ecg['subject'].unique()):
    #     print('\n!!! There is more participants in the dataset with only ECGs !!!')
    #     print('This should not be the case.')
    #     print('Please inspect what is happening in a code editor (spyder..), or ask Yvan.')
    #     more_sub = set(df_full_ecg['subject'].unique()) - set(df_full['subject'].unique())
    #     df_more = df_full_ecg[df_full_ecg['subject'].isin(more_sub)]
    #     df_more.to_csv(f'{summary_path}/ECG_toomany_edf.csv', sep = '\t')
    #     print(f'Saving informations from participants suspect ECGs to:\n{summary_path}/ECG_toomany_edf.tsv')
    
    # # saving info from ECG
    # df_full_ecg.to_csv(f'{summary_path}/ECG_summary.tsv', sep = '\t')
    # print(f'\nSaving informations from ECGs to:\n{summary_path}/ECG_summary.tsv')

    # print('<h2"><b>ECG:</b></h2>', file=f)
    # if not df_full_ecg.empty:
    #     print('<div class="indent1">', file=f)
        
    #     # ECG configuration check______________________________________________
    #     ecg_per_sub = df_full_ecg.groupby('subject')['channel'].apply(lambda x: tuple(sorted(set(x))))

    #     # identify the channel configuration of each participant and store them in a dict to print per channel config
    #     ecg_config_dict = {}
    #     for config in ecg_per_sub.unique():
    #         sub = ecg_per_sub[ecg_per_sub == config].index.tolist()
    #         ecg_config_dict[config] = sub
        
    #     if len(ecg_config_dict) > 1:
    #         print(f'<p>⚠️ {len(ecg_config_dict)} different ECG configurations found</p>', file=f)
    #         print('<p class="indent2">You will have to harmonize the number and the name of channels for your analysis</p>', file=f)
    #         print('<p class="indent2"><i>-WIP- You can use the notebook "TBA" to do it -WIP-</i></p>', file=f)
    #     else:
    #         print('<p>✅ All your participants have the same ECG channels</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG sampling frequency check_________________________________________
    #     # the sampling frequency configuration
    #     sfecg_per_sub = df_full_ecg.groupby('subject')['sampling_frequency'].apply(lambda x: tuple(sorted(set(x))))
    #     # identify the sampling frequency configuration of each participant and store them in a dict to print per sampling configuration config
    #     sfecg_config_dict = {}
    #     for config in sfecg_per_sub.unique():
    #         sub = sfecg_per_sub[sfecg_per_sub == config].index.tolist()
    #         sfecg_config_dict[config] = sub
        
    #     if len(sfecg_config_dict) > 1:
    #         print(f'<p>⚠️ {len(sfecg_config_dict)} different ECG sampling frequencies found: {join_uniq(df_full_ecg["sampling_frequency"])} Hz</p>', file=f)
    #         print('<p class="indent2">You can either re-export your data or downsample to a common sampling frequency for your analysis</p>', file=f)
    #         print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported</i></p>', file=f)
    #     else:
    #         print(f'<p>✅ All your participants have the same ECG sampling frequency: {join_uniq(df_full_ecg["sampling_frequency"])} Hz</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG filters check____________________________________________________
    #     # Get the list of participants with different filtering parameters
    #     # 1st replace NaN because groupby does not like NaN
    #     df_ecgfilt = df_full_ecg.copy()
    #     df_ecgfilt[['lowpass', 'highpass', 'notch']] = df_ecgfilt[['lowpass', 'highpass', 'notch']].fillna('missing')
        
    #     config_ecgfilters = (
    #         df_ecgfilt.groupby(['lowpass', 'highpass', 'notch'])['subject']
    #         .apply(lambda x: sorted(set(x)))
    #         .reset_index(name = 'subjects')
    #     )
        
    #     if len(config_ecgfilters) > 1:
    #         print(f'<p>⚠️ {len(config_ecgfilters)} different ECG filters configurations found:</p>', file=f)
    #         for idx, row in config_ecgfilters.iterrows():
    #             print(f'<p class="indent3">filter config. {idx+1}: hp = {row["highpass"]} Hz; lp = {row["lowpass"]} Hz; notch = {row["notch"]} Hz<br></p>', file=f)
    #         print('<p class="indent2">You can either re-export your data or filter your data to a common fequency</p>', file=f)
    #         print('<p class="indent2"><i>You can use the notebook "inspect_edf_voila.ipynb" to identify which participants need to be re-exported or filtered</i></p>', file=f)
    #     else:
    #         print(f'<p>✅ All your participants have the same ECG filters: hp = {join_uniq(config_ecgfilters["highpass"])} Hz; lp = {join_uniq(config_ecgfilters["lowpass"])} Hz; notch = {join_uniq(config_ecgfilters["notch"])} Hz</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG units check______________________________________________________
    #     if len(df_full_ecg['dimension'].unique()) > 1:
    #         print(f'<p>⚠️ {len(df_full_ecg["dimension"].unique())} different ECG units found: {join_uniq(df_full_ecg["dimension"])}</p>', file=f)
    #         print('<p class="indent2">Before analyzing, make sure that your software (MNE, FIELDTRIP) correctly read your data unit</p>', file=f)
    #     else:
    #         print(f'<p>✅ All your participants have the same ECG unit: {join_uniq(df_full_ecg["dimension"])}</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG inversion check__________________________________________________
    #     df_full_ecginv = df_full_ecg[df_full_ecg['physical_min'] > df_full_ecg['physical_max']]
    #     if not df_full_ecginv.empty:
    #         print('<p><b>ECG polarity</b>: ❌ ECGs with inverted polarity detected!</p>', file=f)
    #         print(f'<p class="indent2">It concerns files: {join_uniq(df_full_ecginv["subject"])}</p>', file=f)
    #         print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
    #         df_full_ecginv.to_csv(f'{summary_path}/ECG_inverted_polarity.tsv', sep = '\t')
    #         print(f'\nSaving informations from inverted polarity ECGs to:\n{summary_path}/ECG_inverted_polarity.tsv')
    #     else:
    #         print('<p><b>ECG polarity</b>: ✅ no inverted polarity detected in ECGs!</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG clipping check___________________________________________________
    #     dr_ecgmask = df_full_ecg['res_theoretical']*pow(2,16) <= dr_thres
    #     bad_ecgdr = df_full_ecg[dr_ecgmask]
    #     if not bad_ecgdr.empty:
    #         print(f'<p><b>ECG clipping</b>: ❌ ECGs with clipping (dynamic range <= {dr_thres} µV) detected!</p>', file=f)
    #         print(f'<p class="indent2">It concerns files: {join_uniq(bad_ecgdr["subject"])}</p>', file=f)
    #         print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
    #         bad_ecgdr.to_csv(f'{summary_path}/ECG_bad_dynamic_range.tsv', sep = '\t')
    #         print(f'\nSaving informations from bad dynamic range ECGs to:\n{summary_path}/ECG_bad_dynamic_range.tsv')
        
    #     else:
    #         print(f'<p><b>ECG clipping</b>: ✅ no clipping detected in ECGs (dynamic range <= {dr_thres} µV)!</p>', file=f)
    #     #______________________________________________________________________
        
    #     # ECG resolution check_________________________________________________
    #     r_ecgmask = df_full_ecg['res_theoretical'] >= r_thres
    #     bad_ecgres = df_full_ecg[r_ecgmask]
    #     if not bad_ecgres.empty:
    #         print(f'<p><b>ECG resolution</b>: ❌ ECGs with low resolution (>= {r_thres} µV) detected!</p>', file=f)
    #         print(f'<p class="indent2">It concerns files: {join_uniq(bad_ecgres["subject"])}</p>', file=f)
    #         print('<p class="indent2"><b>We strongly recommend to re-export the data</b></p>', file=f)
    #         bad_ecgres.to_csv(f'{summary_path}/ECG_bad_resolution.tsv', sep = '\t')
    #         print(f'\nSaving informations from bad resolution ECGs to:\n{summary_path}/ECG_bad_resolution.tsv')
    #     else:
    #         print(f'<p><b>ECG resolution</b>: ✅ no low resolution detected in ECGs (>= {r_thres} µV)!</p>', file=f)
    #     #______________________________________________________________________
    #     print('</div>', file=f)
    #     # print('<p class="indent1">✅ Extraction of ECG parameters completed!</p>', file=f)
    # else:
    #     print('<p class="indent1">❌ No ECG found </p>', file=f)
    ###########################################################################
    print("</body></html>", file=f)
    ###########################################################################
print("\n----------------------------------------------------------------------")
print("----------------------- End of the inspection ------------------------")
print("----------------------------------------------------------------------")

















