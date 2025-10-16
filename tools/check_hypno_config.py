# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:17:36 2025

Check hypnogram configurations within a database.

@author: yvan.nedelec
"""
#%% import packages and define datapath
import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Définition du chemin des données
path_data = '/Users/yvan.nedelec/OneDrive - ICM/Documents/RE_yvan/projects/noemie_rescue/data'

#%% Check if you have a matching number of edf and hypnogram
# Recherche des fichiers EDF
edf_files = glob.glob(os.path.join(path_data, '*.edf'))

# définition du suffixe des fichiers d'hypnogramme - il faut que le nom des hypno soit consistent, ou alors utilisé simplement une recherche de fichier avec '*.txt' dans le dossier de données (si et seulement si il n'y a pas d'autre fichier texte dans la base de données)
hypno_suffix = "_Hypnogram_Export.txt"

# recherche de fichier hypnogramme
hypno_files = []
for root, _, files in os.walk(path_data):
    for fname in files:
        fullpath = os.path.join(root, fname)
        if not os.path.isfile(fullpath):    # ignore path that are not files
            continue
        if fname.startswith('.'):           # ignore path that are starting with "."
            continue
        if hypno_suffix in fname:
            hypno_files.append(os.path.join(path_data, fname))

if len(hypno_files) < len(edf_files): print('You have less hypnograms than .edf files! You are probably missing hypnograms')

if len(hypno_files) > len(edf_files): print('You have more hypnograms than .edf files! Something went wrong in the process')

if len(hypno_files) == len(edf_files): print('Numbers of hypnograms and .edf files are matching!')

#%% extract hypno configurations
hypno_df = pd.DataFrame()
hypno_list = []
# ID_list = []
failed_hypno = []
idx_interro = pd.DataFrame()

# select one file to debug or test features____________________________________
# sub_file = "15_N2"
# pattern = re.compile(rf"(?<!\d){re.escape(sub_file)}") 
# mask_file = np.array([bool(pattern.search(f)) for f in hypno_files])
# file = hypno_files[np.where(mask_file)[0][0]]
#______________________________________________________________________________

for file in hypno_files: # change files to missing_files if you want to compute only the missing files
    file_ID = Path(file).stem.removesuffix(Path(hypno_suffix).stem)  # Extrait "15_N1"
    
    try: hypno = np.loadtxt(file, dtype=str).astype('<U10')
    except Exception as e:
        print(f"⚠️ Erreur chargement hypnogramme pour {file_ID} : {e}")
        failed_hypno.append((file, 'hypno loading'))
        continue
    
    cur_df = pd.DataFrame({"hypno": [np.array2string(np.unique(hypno))], "sub_ID": [file_ID]}) # converting the array of unique values of hypno to string because I had an error TypeError: unhashable type: 'numpy.ndarray' if I was using the array directly 
    hypno_df = pd.concat([hypno_df, cur_df], ignore_index=True)
    
    cur_idx_interro = pd.DataFrame({"idx_?": np.where(hypno == "?")[0], "hypno_length": len(hypno), "sub_ID": file_ID})
    idx_interro = pd.concat([idx_interro, cur_idx_interro], ignore_index=True)
    hypno_list.append(hypno)

print(f"You have {len(hypno_df['hypno'].unique())} different configurations of hypnogram\n")    
for c, config in enumerate(hypno_df["hypno"].unique()):
    df_hyp = hypno_df[hypno_df['hypno'] == config].copy()
    print(f"\n #{c+1} configuration ({len(df_hyp['sub_ID'].unique())} participants):")
    print(f"{config}: \n{df_hyp['sub_ID'].unique()}")

#%% Print a warning if the "?" in the hypnogram is not at the boundaries fo the recording (within 10 first or 10 last epochs)
issue = []
for idx, row in idx_interro.iterrows():
    if row["idx_?"] > 10 and row["idx_?"] < row["hypno_length"]-10:
        print(f'{row["sub_ID"]} might have a sleep scoring issue. \nAn "?" have been found outside of the boundaries of the night, at the epoch {row["idx_?"]}/{row["hypno_length"]}')
        issue.append(idx)

if not issue:
    print('there is no issue with label "?" in the hypnograms')
    
#%% remap hypnograms labels (but make sure that there is no "?" issue, because ? will be changed to W)
remapped_df = pd.DataFrame()
remapped_list = []

for file in hypno_files: # change hypno_files to missing_files if you want to compute only the missing files
    file_ID = Path(file).stem.removesuffix(Path(hypno_suffix).stem)  # Extrait "15_N1"
    
    try: hypno = np.loadtxt(file, dtype=str).astype('<U10')
    except Exception as e:
        print(f"⚠️ Erreur chargement hypnogramme pour {file_ID} : {e}")
        failed_hypno.append((file, 'hypno loading'))
        continue

    remapped_hypno = hypno.copy()
    
    if np.any(np.isin(remapped_hypno, "?")):
        remapped_hypno = np.strings.replace(remapped_hypno, '?', 'W').astype('<U10')
    
    if not np.any(np.isin(remapped_hypno, ['N1', 'N2', 'N3'])):
        remapped_hypno = np.strings.replace(remapped_hypno, '1', 'N1')
        remapped_hypno = np.strings.replace(remapped_hypno, '2', 'N2')
        remapped_hypno = np.strings.replace(remapped_hypno, '3', 'N3')
        remapped_hypno = np.strings.replace(remapped_hypno, '4', 'N3')
        
    cur_remapped = pd.DataFrame({"hypno": [np.array2string(np.unique(remapped_hypno))], "sub_ID": [file_ID]}) # converting the array of unique values of hypno to string because I had an error TypeError: unhashable type: 'numpy.ndarray' if I was using the array directly 
    remapped_df = pd.concat([remapped_df, cur_remapped], ignore_index=True)
    remapped_list.append(remapped_hypno)

# recheck the configuration
print(f"After remapping, you have {len(remapped_df['hypno'].unique())} different configurations of hypnogram\n")    
for c, config in enumerate(remapped_df["hypno"].unique()):
    df_hyp = remapped_df[remapped_df['hypno'] == config].copy()
    print(f"\n #{c+1} configuration ({len(df_hyp['sub_ID'].unique())} participants):")
    print(f"{config}: \n{df_hyp['sub_ID'].unique()}")
    
#%% check if there is any unexepected symbols in the remapped hypno
unexpect_symb = pd.DataFrame() # init a list to store the not expected symbols
for h, hyp in enumerate(remapped_list):
    sub_ID = remapped_df.iloc[h]["sub_ID"]
    for cur_code in np.unique(hyp): 
        if cur_code not in ['W', 'N1', 'N2', 'N3', 'R']:
            cur_df = pd.DataFrame({"unexpected_symbols": cur_code, "sub_ID": sub_ID})
            unexpect_symb = pd.concat([unexpect_symb, cur_df], ignore_index = True)

if not unexpect_symb.empty:
    print("Unexpected symboles in the remapped hypnograms:")
    for p, participant in unexpect_symb['sub_ID'].unique():
        p_df = unexpect_symb[unexpect_symb['sub_ID'] == participant].copy()
        print(f"Participant {participant} have unexpected symbols: {p_df['unexpected_symbols'].unique().astype(str).tolist()}")
else:
    print("There is no unexpected symbols in your remapped hypnograms")

#%% save the remapping 
for h, hyp in enumerate(remapped_list):
    sub_ID = remapped_df.iloc[h]["sub_ID"]
    remapped_suffix = "_Hypnogram_remapped.txt"
    save_path = os.path.join(path_data, sub_ID+remapped_suffix)
    np.savetxt(save_path, hyp, fmt='%s')

