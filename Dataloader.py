# import 
import os
import re
import datetime
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd 
from glob import glob
import re
from pandas import ExcelWriter
import math
from math import sqrt, acos, atan2, sin, cos
from collections import defaultdict
import pyxdf
import neurokit2 as nk

import statistics
import scipy
import pymannkendall as mk
import matplotlib.pyplot as plt 
import seaborn as sns
import bioread

import missingno as msno
import csv
import time
from src.utils import (
    time2frame, metadata2frame, coughxlsx2df, processed2df, cut2case, cut2single, 
    _make_query, make_queries, query, query_w, getqueries, tval_df, bubbles2e, 
    createFolder, data2csv, save2pkl
)

from src.utils_plot import(
    cutplot, draw_corr_pval, drawheatmap_ttest,
)

def xdfreaderfixer(directory: str, infodir: str = "ShimmerData.csv"):
    streams, header = pyxdf.load_xdf(directory)
    descs = {}

    index = 0
    for stream in streams:
        if stream["info"]['channel_count'][0]  == "2":
            if stream["info"]["name"][0] != "Unity.Transform_Scene":
                desc_p = defaultdict(list)
                acquisition_p = defaultdict(list)
                channels_p = defaultdict(list)
                channame_p = ["Slider", "Timestamp"]
                channel_p = []
                for item in channame_p:
                    dict = defaultdict(list)
                    dict['label'] = [item]
                    channel_p.append(dict)
                channels_p['channel'] = channel_p
                acquisition_p['manufacturer'] = ['psychopy']
                desc_p['acquisition'] = [acquisition_p]
                desc_p['channels'] = [channels_p]
                stream["info"]['desc'][0] = desc_p
                descs[index] = desc_p
            
            
        else:
            data = pd.read_csv(infodir)
            chantype = data.loc[1][:-1].values.flatten().tolist()
            names = data.loc[0][:-1].values.flatten().tolist()
            channame = []

            for idx in range(len(chantype)):
                channame.append(names[idx]+"_"+chantype[idx])
                
            desclist = []
            desc = defaultdict(list)
            acquisition = defaultdict(list)
            channels = defaultdict(list)
            channel = []
            for item in channame:
                dict = defaultdict(list)
                dict['label'] = [item]
                channel.append(dict)

            channels['channel'] = channel
            acquisition['manufacturer'] = ['Shimmer3']
            desc['acquisition'] = [acquisition]
            desc['channels'] = [channels]
            stream["info"]['desc'][0] = desc
            descs[index] = desc
        
        index += 1
            
    testdata = nk.read_xdf(directory, desc = descs, upsample= 1)
    
    return testdata

def rename_duplicates(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(f"{col}")
    return new_columns

import os
import traceback  # To capture full error details
from glob import glob
import pandas as pd
import neurokit2 as nk

def dataloader(datapath_top: str, scenes: list):
    '''
    This function loads experiment data while continuing even if errors occur.
    
    ✅ Logs which participants cause errors
    ✅ Identifies which function failed
    ✅ Outputs problematic data for debugging
    '''

    names = []
    data = {
        "Psychopy": [],
        "Physiology": [],
        "Tracker": [],
        "Eyetracker": [],
        "Facetracker": []
    }
    unity = defaultdict(dict)
    zeros = defaultdict(dict)
    pIDs = defaultdict(dict)

    for folder in os.listdir(datapath_top):        
        datapath = os.path.join(datapath_top, folder)
        names.append(folder)
        print(f"\n🔹 Processing participant: {folder}")

        try:
            # ✅ 1) Process Anxiety Log Files
            for log in glob(datapath + '/*.log'):
                try:
                    data["Psychopy"].append(log)
                except Exception as e:
                    print(f"❌ Error in Processing Anxiety Log ({folder})\n{e}")
                    traceback.print_exc()

            # ✅ 2) Process Physiology, Tracker, and Unity Data
            for in_folder in os.listdir(datapath):
                file_path = os.path.join(datapath, in_folder)
                file_ext = in_folder.split(".")[-1]

                # ✅ Read .acq files (Physiology)
                if file_ext == "acq":
                    try:
                        _, _, _, result = read_acqknowledge_with_markers(file_path)
                        result['Subject'] = folder
                        data["Physiology"].append(result)
                    except Exception as e:
                        print(f"❌ Error in read_acqknowledge_with_markers() ({folder}) - File: {in_folder}\n{e}")
                        traceback.print_exc()

                # ✅ Read .xdf files (Tracker)
                elif file_ext == "xdf":
                    try:
                        dtype = "Tracker"
                        data[dtype].append(xdfreaderfixer(file_path))
                    except Exception as e:
                        print(f"❌ Error in xdfreaderfixer() ({folder}) - File: {in_folder}\n{e}")
                        traceback.print_exc()

                # ✅ Read CSV Files (Unity Data)
                elif file_ext == "csv":
                    try:
                        scene = in_folder.split("_")[0]
                        dtype = in_folder.split("_")[1].split(".")[0]

                        if dtype != "SUDS":
                            _df = read_csv_with_max_columns(file_path)  # ✅ Use optimized function

                            if dtype == "position":
                                zeros[scene][dtype] = _df[' Time'][0]

                            if dtype in ["position", "rotation"]:
                                _df = _df.rename(columns={" Time": "Time", " X": "X", " Y": "Y", " Z": "Z"})

                            _df['Subject'] = folder
                            unity[dtype][scene] = _df
                    except Exception as e:
                        print(f"❌ Error in Reading CSV ({folder}) - File: {in_folder}\n{e}")
                        traceback.print_exc()

            # ✅ 3) Process Tracker Data into Eye/Face Tracker
            try:
                df = data['Tracker'][0][0]
                df.columns = rename_duplicates(list(df.columns))

                mandatory_columns = ['Scene', 'UnityTime']
                unit_columns = [col for col in df.columns if 'unit' in col]
                facetracker = df[mandatory_columns + unit_columns]
                other_columns = [col for col in df.columns if col not in mandatory_columns + unit_columns]
                eyetracker = df[mandatory_columns + other_columns]

                mapping = {0: "Start", 1: "Practice", 2: "ElevatorTest", 3: "Elevator1",
                           4: "Outside", 5: "Hallway", 6: "Elevator2", 7: "Hall", 8: "End"}
                
                eyetracker = eyetracker.copy()
                facetracker = facetracker.copy()

                eyetracker.loc[:, 'Scene'] = eyetracker['Scene'].map(mapping)
                facetracker.loc[:, 'Scene'] = facetracker['Scene'].map(mapping)
                eyetracker = eyetracker.rename(columns={"Scene": "scene"})
                facetracker = facetracker.rename(columns={"Scene": "scene"})
                eyetracker.loc[:, 'Subject'] = folder
                facetracker.loc[:, 'Subject'] = folder

                data["Eyetracker"].append(eyetracker)
                data["Facetracker"].append(facetracker)

            except Exception as e:
                print(f"❌ Error in Processing Tracker Data ({folder})\n{e}")
                traceback.print_exc()

        except Exception as e:
            print(f"\n🚨 Critical Error in Participant {folder}: {e}")
            traceback.print_exc()

    return names, data, unity

# def dataloader(datapath_top: str, scenes: list):
#     '''
#     This function load data from experiment file folder.
#     Data list
#     #Anxiety data(.log)-anxiety_psy
#     #Transform data(.csv)-position, rotation, customevent in/out
#     '''
#     #Data loader: Position and rotation data, anxiety data.
#     #folder name
#     names = []
#     #Anxiety data psychopy room
#     data = {}
#     data["Psychopy"] = []
#     #Physiology data room
#     data["Physiology"] = []
#     data["Tracker"] = []
#     data["Eyetracker"] = []
#     data["Facetracker"] = []
    
    
#     #Transform data
#     unity = defaultdict(dict)
#     #start time
#     zeros = defaultdict(dict)
#     pIDs = defaultdict(dict)
    
#     ## Create List for the Space tp Store
#     datapath = os.path.join(datapath_top, os.listdir(datapath_top)[0])
#     for in_folder in os.listdir(datapath):
#         formet = in_folder.split(".")[-1]
#         # if formet == "xdf":
#         #     dtype = in_folder.split("_")[-1].split(".")[0]
#         #     data[dtype] = []
#         if formet == "csv":
#             scene = in_folder.split("_")[0]
#             dtype = in_folder.split("_")[1].split(".")[0]
#             unity[dtype][scene] = []
#             pIDs[dtype][scene] = []
            
   
#     for folder in os.listdir(datapath_top):        
#         datapath = os.path.join(datapath_top, folder)
#         names.append(folder)
#         print(folder)
#         #anxiety data
#         for log in glob(datapath+ '/*.log'):
#             data["Psychopy"].append(log)
            
#         #Transform data
#         _allpos = pd.DataFrame()
#         _allrot = pd.DataFrame()
#         _allcus = pd.DataFrame()
        
#         for in_folder in os.listdir(datapath):
#             formet = in_folder.split(".")[-1]
#             if formet == "acq":
#                 dtype = in_folder.split("_")[-1].split(".")[0]
#                 if dtype == "VR":
#                     _,_,_,result = read_acqknowledge_with_markers(os.path.join(datapath, in_folder))
#                     result['Subject'] = folder
#                     data["Physiology"].append(result)
#             if formet == "xdf":
#                 dtype = "Tracker"
#                 data[dtype].append(xdfreaderfixer(os.path.join(datapath, in_folder)))
#             if formet == "csv":
#                 scene = in_folder.split("_")[0]
#                 dtype = in_folder.split("_")[1].split(".")[0]
#                 if dtype != "SUDS":
#                     if dtype == "customevent":
#                         _df = pd.read_csv(datapath+'/'+in_folder,
#                                 engine='python',
#                                 encoding='utf-8',
#                                 names = ['ID','Time','Action', 'Actor', '1','2','3','4']
#                                 ,header = None)   
#                         _df = _df.iloc[1:, :] 
                        

#                     else:
#                         _df = pd.read_csv(datapath+'/'+in_folder,
#                                 engine='python',
#                                 encoding='utf-8')
#                     if dtype == "position":
#                         zeros[scene][dtype] = _df[' Time'][0] 
                    
#                     if (dtype == "position") | (dtype == "rotation"):
#                         _df = _df.rename(columns={" Time": "Time"," X": "X"," Y": "Y"," Z": "Z"})
                        
#                     _df['Subject'] = folder

#                     # print(dtype, scene)
#                     unity[dtype][scene].append(_df)
    
        
#         df = data['Tracker'][0][0]
#         # Now split the dataframe
#         mandatory_columns = ['Scene', 'UnityTime']
#         df.columns = rename_duplicates(list(df.columns))
#         # Identify "unit" columns
#         unit_columns = [col for col in df.columns if 'unit' in col]
#         # First dataframe: mandatory columns + "unit" columns
#         facetracker= df[mandatory_columns + unit_columns]
#         # Second dataframe: mandatory columns + all other columns not in df1
#         other_columns = [col for col in df.columns if col not in mandatory_columns + unit_columns]
#         eyetracker = df[mandatory_columns + other_columns]
        
#         mapping = {
#         0: "Start",
#         1: "Practice",
#         2: "ElevatorTest",
#         3: "Elevator1",
#         4: "Outside",
#         5: "Hallway",
#         6: "Elevator2",
#         7: "Hall",
#         8: "End"
#         }

#         # Replace values in the column
#         eyetracker['Scene'] = eyetracker['Scene'].map(mapping)
#         facetracker['Scene'] = facetracker['Scene'].map(mapping)
#         eyetracker.rename(columns={"Scene": "scene"}, inplace=True)
#         facetracker.rename(columns={"Scene": "scene"}, inplace=True)
#         eyetracker['Subject'], facetracker['Subject'] = folder, folder
        
#         data["Eyetracker"].append(eyetracker)
#         data["Facetracker"].append(facetracker)

                

#     return (names, data, unity)

def read_csv_with_max_columns(file_path, encoding='utf-8'):
    """
    Efficiently determine the maximum number of columns in a CSV and load it dynamically.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file.
    encoding : str
        Encoding of the file (default: 'utf-8').

    Returns:
    ----------
    df : pandas.DataFrame
        DataFrame with the correct number of columns.
    """
    try:
        # Step 1: Determine the maximum number of columns
        max_columns = 0
        all_rows = []

        with open(file_path, "r", encoding=encoding) as f:
            reader = csv.reader(f)
            for row in reader:
                max_columns = max(max_columns, len(row))  # Track the max columns
                all_rows.append(row)  # Store rows for processing

        # Step 2: Normalize rows to have the same number of columns
        normalized_rows = [row + [''] * (max_columns - len(row)) for row in all_rows]

        # Step 3: Create a DataFrame from the normalized rows
        df = pd.DataFrame(normalized_rows)

        # Step 4: Use the first row as column names (if applicable)
        df.columns = df.iloc[0]  # Set column headers
        df = df[1:].reset_index(drop=True)  # Drop the header row from data

        return df

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

###################.acq 파일 마커와 같이 전처리 #################
import os
from collections import Counter
import numpy as np
import pandas as pd
from neurokit2.signal import signal_resample


import os
from collections import Counter
import numpy as np
import pandas as pd
from neurokit2.signal import signal_resample
import sys
import contextlib

def read_acqknowledge_with_markers(filename, sampling_rate="max", resample_method="interpolation", impute_missing=True):
    """
    Read and process a BIOPAC AcqKnowledge file (.acq) while handling VR, Survey, and BT files differently.

    Parameters
    ----------
    filename : str
        Path to a BIOPAC's AcqKnowledge (.acq) file.
    sampling_rate : int or "max"
        Desired sampling rate in Hz. "max" uses the highest recorded rate.
    resample_method : str
        Method for resampling signals.
    impute_missing : bool
        Whether to impute missing values in signals.

    Returns
    ----------
    df : DataFrame
        Processed signal data from the .acq file.
    event_markers_df : DataFrame
        Event markers with ['Sample', 'scene', 'marker'].
    sampling_rate : int
        The actual sampling rate used.
    result : DataFrame
        Merged dataframe of signal and event markers.
    """

    # Scene mapping (moved inside function)
    scene_mapping = {
        1: "Practice",
        2: "ElevatorTest",
        3: "Elevator1",
        4: "Outside",
        5: "Hallway",
        6: "Elevator2",
        7: "Hall"
    }

    # Check if file exists
    if not os.path.exists(filename):
        raise ValueError(f"File not found: {filename}")

    # Read AcqKnowledge file
    # Silence bioread warnings
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            file = bioread.read_file(filename)  # Read file silently

    # Determine max sampling rate
    if sampling_rate == "max":
        sampling_rate = max(channel.samples_per_second for channel in file.channels)

    # Process data channels
    data = {}
    channel_counter = Counter()
    for channel in file.channels:
        signal = np.array(channel.data)

        # Handle missing data
        if impute_missing and np.isnan(signal).any():
            signal = pd.Series(signal).fillna(method="pad").values

        # Resample signal if needed
        if channel.samples_per_second != sampling_rate:
            signal = signal_resample(
                signal,
                sampling_rate=channel.samples_per_second,
                desired_sampling_rate=sampling_rate,
                method=resample_method,
            )

        # Handle duplicate channel names
        channel_name = channel.name
        if channel_counter[channel_name] > 0:
            channel_name = f"{channel_name} ({channel_counter[channel_name]})"
        data[channel_name] = signal
        channel_counter[channel.name] += 1

    # Align signal lengths
    max_length = max(len(signal) for signal in data.values())
    for channel_name, signal in data.items():
        if len(signal) < max_length:
            data[channel_name] = np.pad(signal, (0, max_length - len(signal)), mode="edge")

    # Create DataFrame for signal data
    df = pd.DataFrame(data)

    # Extract event markers
    event_markers = []
    for marker in file.event_markers:
        event_markers.append({
            "Time (s)": marker.sample_index / sampling_rate,  # Convert index to seconds
            "Text": marker.text
        })

    # Convert to DataFrame
    event_markers_df = pd.DataFrame(event_markers)
    event_markers_df['Sample'] = (event_markers_df['Time (s)'] * sampling_rate).astype(int)

    # Detect file type
    file_basename = os.path.basename(filename)
    is_vr = "_VR.acq" in file_basename
    is_survey = "_Survey.acq" in file_basename
    is_bt = "_BT.acq" in file_basename

    # Process VR files (Use number-based mapping)
    if is_vr:
        event_markers_df[['scene_number', 'marker']] = event_markers_df['Text'].str.split(',', expand=True).iloc[:, :2]
        event_markers_df['scene_number'] = pd.to_numeric(event_markers_df['scene_number'], errors='coerce')
        event_markers_df['scene'] = event_markers_df['scene_number'].map(scene_mapping)
        event_markers_df.drop(columns=['scene_number'], inplace=True)
        event_markers_df.set_index("Sample", inplace=True)
        # Assign missing markers
        for i in range(1, 8):
            startp_rows = event_markers_df[(event_markers_df['scene'] == scene_mapping.get(i)) & (event_markers_df['marker'] == "StartP")]
            end_rows = event_markers_df[(event_markers_df['scene'] == scene_mapping.get(i)) & (event_markers_df['marker'] == "End")]
            
            if not startp_rows.empty and not end_rows.empty:
                startp_idx = startp_rows.index[0]
                end_idx = end_rows.index[0]
                df.loc[(df.index >= startp_idx) & (df.index <= end_idx), "marker"] = "Ongoing"
                df.loc[(df.index >= startp_idx) & (df.index <= end_idx), "scene"] = scene_mapping.get(i)

    # Process Survey files (Use scene name directly)
    elif is_survey:
        event_markers_df[['scene_number', 'scene']] = event_markers_df['Text'].str.split('_', n=1, expand=True)
        event_markers_df.drop(columns=['scene_number'], inplace=True)

    # Process BT files (Extract bt images)
    elif is_bt:
        event_markers_df['bt_image'] = event_markers_df['Text'].apply(lambda x: x if 'bt' in x else np.nan)

    # Drop unnecessary text column
    event_markers_df.drop(columns=['Text'], inplace=True)

    # # # Set index as Sample for merging with signal data
    # # event_markers_df.set_index('Sample', inplace=True)

    # # Merge signal data with event markers
    # result = pd.concat([df, event_markers_df], axis=1)

    return df, event_markers_df, sampling_rate

# def read_acqknowledge_with_markers(filename, sampling_rate="max", resample_method="interpolation", impute_missing=True):
#     """
#     Read and format a BIOPAC's AcqKnowledge file into a pandas' dataframe, including event markers.

#     Parameters
#     ----------
#     filename : str
#         Filename (with or without the extension) of a BIOPAC's AcqKnowledge file (e.g., "data.acq").
#     sampling_rate : int or "max"
#         Desired sampling rate in Hz. "max" uses the maximum recorded sampling rate.
#     resample_method : str
#         Method of resampling.
#     impute_missing : bool
#         Whether to impute missing values in the signal.

#     Returns
#     ----------
#     df : DataFrame
#         The AcqKnowledge file as a pandas dataframe.
#     event_markers : DataFrame
#         Event markers with columns ['Time (s)', 'Channel', 'Type', 'Text'].
#     sampling_rate : int
#         Sampling rate used in the data.

#     """
#     try:
#         import bioread
#     except ImportError:
#         raise ImportError("Please install the 'bioread' module (`pip install bioread`).")

#     # Check filename
#     if not filename.endswith(".acq"):
#         filename += ".acq"

#     if not os.path.exists(filename):
#         raise ValueError(f"File not found: {filename}")

#     # Read the AcqKnowledge file
#     file = bioread.read_file(filename)

#     # Determine sampling rate
#     if sampling_rate == "max":
#         sampling_rate = max(channel.samples_per_second for channel in file.channels)

#     # Process data channels
#     data = {}
#     channel_counter = Counter()
#     for channel in file.channels:
#         signal = np.array(channel.data)

#         # Handle missing data
#         if impute_missing and np.isnan(signal).any():
#             signal = pd.Series(signal).fillna(method="pad").values

#         # Resample signal
#         if channel.samples_per_second != sampling_rate:
#             signal = signal_resample(
#                 signal,
#                 sampling_rate=channel.samples_per_second,
#                 desired_sampling_rate=sampling_rate,
#                 method=resample_method,
#             )

#         # Handle duplicate channel names
#         channel_name = channel.name
#         if channel_counter[channel_name] > 0:
#             channel_name = f"{channel_name} ({channel_counter[channel_name]})"
#         data[channel_name] = signal
#         channel_counter[channel.name] += 1

#     # Align signal lengths
#     max_length = max(len(signal) for signal in data.values())
#     for channel_name, signal in data.items():
#         if len(signal) < max_length:
#             data[channel_name] = np.pad(signal, (0, max_length - len(signal)), mode="edge")

#     # Create DataFrame for signal data
#     df = pd.DataFrame(data)

#     # Extract event markers
#     event_markers = []
#     for marker in file.event_markers:
#         event_markers.append({
#             "Time (s)": marker.sample_index / sampling_rate,
#             "Channel": marker.channel_name,
#             "Type": marker.type,
#             "Text": marker.text
#         })
#     event_markers_df = pd.DataFrame(event_markers)
#     event_markers_df['Sample'] = event_markers_df['Time (s)']*2000
#     event_markers_df[['scene', 'marker']] = event_markers_df['Text'].str.split(',', n=2, expand=True)[[0, 1]]
#     event_markers_df = event_markers_df[event_markers_df['Type'] != 'Append']
#     event_markers_df.drop(columns=['Text','Channel','Type','Time (s)'], inplace=True)
#     event_markers_df['scene'] = pd.to_numeric(event_markers_df['scene'])
#     event_markers_df['Sample'] = event_markers_df['Sample'].apply(np.int64)
#     event_markers_df.set_index('Sample', inplace=True)
#     result = pd.concat([df, event_markers_df], axis=1)
#     for i in range(1,8):
#         # startidx = result[(result['scene']==i)&(result['marker']=="Start")].index[0]
#         startpidx = result[(result['scene']==i)&(result['marker']=="StartP")].index[0]
#         endidx = result[(result['scene']==i)&(result['marker']=="End")].index[0]
#         result.loc[startpidx:endidx, 'scene'] = i
#         result.loc[startpidx:endidx, 'marker'] = "Ongoing"
#     result.dropna(inplace=True)
    
    
#     mapping = {
#     1: "Practice",
#     2: "ElevatorTest",
#     3: "Elevator1",
#     4: "Outside",
#     5: "Hallway",
#     6: "Elevator2",
#     7: "Hall"
# }

#     # Replace values in the column
#     result['scene'] = result['scene'].map(mapping)
    
#     ############ Time에 sampling_rate 곱해서 frame 단위로 바꾸고, Time 기본은 남겨두고 frame을 index로 할 것.

#     return df, event_markers_df, sampling_rate, result
def check_acq_markers(file_path):
    """
    Reads a .acq file and prints all marker types found.
    
    :param file_path: Path to the .acq file
    """
    try:
        # Load the AcqKnowledge file
        file = bioread.read_file(file_path)

        if not file.event_markers:
            print(f"⚠️ No event markers found in {file_path}")
            return

        # Print marker details
        print(f"\n🔍 Markers found in {file_path}:")

        for marker in file.event_markers:
            print(f"🔹 Marker Type: {marker.type}, Channel: {marker.channel_name}, Text: {marker.text}")

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")


#####################Anxiety#############################################
# ✅ Updated Anxietyloader
import re
import numpy as np
import pandas as pd

def clip_slider(value):
    return max(0, min(10, value))  # Clip slider between 0 and 10

def Anxietyloader(filepath, name, scenes=["Elevator1", "Outside", "Hallway", "Elevator2", "Hall"]):
    data = []
    scenes_copy = scenes.copy()
    current_scene = None

    with open(filepath, 'r') as file:
        for line in file:
            time_match = re.match(r'(\d+\.\d{4})', line)
            time = float(time_match.group(1)) if time_match else None

            if 'movie: autoDraw = True' in line:
                current_scene = scenes_copy.pop(0) if scenes_copy else None
                data.append({'time': time, 'slider marker': np.nan, 'scene': current_scene, 'name': name})

            elif 'movie: autoDraw = False' in line:
                current_scene = None

            slider_match = re.search(r'slider: markerPos = ([\d\.\-]+)', line)  # 👈 음수도 커버
            if slider_match and current_scene is not None:
                slider_value = clip_slider(float(slider_match.group(1)))
                data.append({'time': time, 'slider marker': slider_value, 'scene': current_scene, 'name': name})

    df = pd.DataFrame(data)

    # Step 1: scene 이름은 ffill (slider 없이 scene만 나오는 경우를 대비)
    df['scene'] = df['scene'].ffill()

    # ❗ Step 2: slider marker는 "절대 fill 안 함" (frame 변환 이후 따로 처리할 것)
    # (bfill, ffill 모두 제거)

    print("✅ Anxietyloader (raw, no fill) processing complete:", filepath)
    return df







# def Anxietyloader(filepath,name, scenes=["Elevator1", "Outside", "Hallway", "Elevator2", "Hall"]):
#     # Initialize variables
    
#     # Initialize variables
    
#     data = []
#     current_scene = None

#     # Define function to clip slider values



#     # Open and process the file
#     with open(filepath, 'r') as file:
#         for line in file:
#             # Extract time
#             time_match = re.match(r'(\d+\.\d{4})', line)
#             time = float(time_match.group(1)) if time_match else None
#             # print(time)

#             # Check for scene changes
#             if 'movie: autoDraw = True' in line:
#                 current_scene = scenes.pop(0) if scenes else None
#                 data.append({'time': time, 'slider marker': np.nan, 'scene': current_scene, 'name': name})
                
#             elif 'movie: autoDraw = False' in line:
#                 current_scene = None

#             # Extract slider marker values
#             slider_match = re.search(r'slider: markerPos = ([\d\.]+)', line)
#             if slider_match:
#                 slider_value = clip_slider(float(slider_match.group(1)))
#                 data.append({'time': time, 'slider marker': slider_value, 'scene': current_scene, 'name': name})

#     # Create DataFrame
#     df = pd.DataFrame(data)
#     df = df.fillna(method="bfill")
#     # Save DataFrame to CSV
#     print("Processing complete")
    
#     return df

# def Anxiety_preprocessing(anxiety: list, scenes: list):
#     '''
#     Preprocessing Anxiety log file data.
#     *파일이 지저분한 관계로 임의 값이 다수 포함되어 있음.
#     설문 코드를 바꾸게 되면 값 수정 필요할 수 있음.
    
#     '''
    
#     anxiety_psy = []

#     for infile in anxiety:
#         name = infile.split("\\")[1]
#         lst = []
#         with open(infile,encoding="ISO-8859-1") as f:
#             f = f.readlines()
            
#         for lines in f:
#             lst.append(re.split(" \t|: | = |\n", lines))


#         anxiety_df = pd.DataFrame(lst, columns=['Time', '1', 'video', 'type', 'marker', 'drop'])
#         anxiety_df = anxiety_df[anxiety_df['1'] == "EXP"]    
#         anxiety_df = anxiety_df.drop(["1","drop"], axis=1)
#         anxiety_df["Scene"] = 0

#         for idx in range(len(scenes)):
#             midx = idx + 1 
#             start = anxiety_df[(anxiety_df['video'] == f"movie{midx}")&(anxiety_df['type'] == "autoDraw")&(anxiety_df['marker'] == "True")].index[0]
#             end = anxiety_df[(anxiety_df['video'] == f"movie{midx}")&(anxiety_df['type'] == "autoDraw")&(anxiety_df['marker'] == "False")].index[0]
#             anxiety_df.loc[start:end+1, "Scene"] = scenes[idx] 
            
#         anxiety_df = anxiety_df[anxiety_df['Scene'] != 0].reset_index(drop=True)
        
#         for scene in scenes:
#             _df = anxiety_df[anxiety_df["Scene"] == scene]
#             first = anxiety_df[anxiety_df["Scene"] == scene].index[0]
#             last = anxiety_df[anxiety_df["Scene"] == scene].index[-1]
#             startval = _df.loc[first+2,"marker"]
#             endval = _df.loc[last-3,"marker"]
            
#             anxiety_df.loc[first:first+2,"marker"] = startval
#             anxiety_df.loc[last-2:last,"marker"] = endval  
            
#         anxiety_df = anxiety_df.drop(["video","type"], axis=1)
#         anxiety_df["Subject"] = name
#         anxiety_df[["Time", "marker"]] = anxiety_df[["Time", "marker"]].astype("float64")
#         anxiety_df["marker"] = np.clip(anxiety_df["marker"],1,5)
#         anxiety_psy.append(anxiety_df)

#     return anxiety_psy


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]


##########################Transform Preprocessing#########################################
def playertransform(position_df, rotation_df):
    '''Player/Agent 를 구분하여 새로운 DF를 만들고 빈 부분을 채움. Rotation value 포함'''
    #Fix column Name-remove unnecessary blank
    position_df.columns  = ['SubjectID', 'Time', 'X', 'Y', 'Z'] # 컬럼명 변경 
    rotation_df.columns  = ['SubjectID', 'Time', 'X', 'Y', 'Z'] # 컬럼명 변경 
    subject_ID_list = position_df['SubjectID'].unique().tolist() #Agent 개체들과 Player 구분을 위한 Subject ID list
    player_ID = position_df['SubjectID'].unique().tolist()[-1] #Player Subject ID

    #Divide into Each Subject ID
    player_pos_df = position_df[position_df['SubjectID'] == player_ID]
    player_rot_df = rotation_df[rotation_df['SubjectID'] == player_ID]
    test_pos_list = list()
    test_rot_list = list()
    all_list = list()
    allinone = list()
    
    
    
    for ID in subject_ID_list:
        #Divide into Each Subject ID
        agent_pos_df = position_df[position_df['SubjectID'] == ID]
        agent_rot_df = rotation_df[rotation_df['SubjectID'] == ID]


        #Index reset
        ##Position
        tmp_player_pos = player_pos_df.reset_index().drop('index',axis=1)
        tmp_agent_pos = agent_pos_df.reset_index().drop('index',axis=1)
        ##Rotation
        tmp_player_rot = player_rot_df.reset_index().drop('index',axis=1)
        tmp_agent_rot = agent_rot_df.reset_index().drop('index',axis=1)
        
        #Drop Duplicated values
        tmp_agent_pos = tmp_agent_pos.drop_duplicates(['Time'], keep='first')
        tmp_agent_rot = tmp_agent_rot.drop_duplicates(['Time'], keep='first')
        

        #Merge into one df
        test_pos = pd.merge(left = tmp_player_pos, right = tmp_agent_pos, how='left', on='Time', suffixes=('_1','_2'))
        test_rot = pd.merge(left = tmp_player_rot, right = tmp_agent_rot, how='left', on='Time', suffixes=('_1','_2'))
 

        #비어있는 Subject ID_2 컬럼은 어차피 ID가 들어가므로 그냥 채워준다.
        test_pos['SubjectID_2'] = ID
        test_rot['SubjectID_2'] = ID
        #비어있는 Subject ID_1 컬럼은 어차피 ID가 들어가므로 그냥 채워준다.
        test_pos['SubjectID_1'] = player_ID
        test_rot['SubjectID_1'] = player_ID
        
        
        test_pos.sort_values(by='Time', axis=0, ascending=True, 
                              inplace=True, kind='quicksort', 
                              na_position='last')
        
     
        allinone = pd.merge(test_pos, test_rot, on='Time', how='outer',suffixes=('_pos','_rot'))
        #print(allinone.count())
        
        
    
        #Fill in missing data 
        allinone = allinone.fillna(method = 'ffill')
 
        allinone = allinone.dropna(axis=0)
        allinone = allinone.reset_index().drop('index',axis=1)
        
        #print(len(allinone))

        all_list.append(allinone)

    return all_list


###############Visual Field Preprocessing ################################
def cos_sim(A, B):
    '''Get Cosine value'''
    return dot(A, B)/(norm(A)*norm(B))    

def getfrontvec(yr):
    '''개체의 정면을 확인함'''
    front_x = 5*np.cos(math.pi * (yr / 180)) #degree to radian. 5 is arbitrary lenth of the vector
    front_z = 5*np.sin(math.pi * (yr / 180))
    front_vec = np.array([front_x,front_z])

    return front_vec

def getvector(px, pz, ax, az):
    '''vector 구하기
    
    도착점 - 시작점
    
    즉 ax, az 좌표가 도착점, px, pz 좌표가 시작점이다.
    
    '''
    vector = np.array([(ax-px),(az-pz)])
    return vector

def visualdegree(vector,front_vec):
    '''Get Visual Degree'''
    degree = np.degrees(np.arccos(cos_sim(vector,front_vec)))
    return degree


def twovecdegree(v1, v2, radian=False):
    '''
    Get degree between two vectors. Output is degree, but if radian = True, you can get radian.
    
    v1, v2 = [float, float] #2d Vector form
    '''
    unitv1 = v1 / np.linalg.norm(v1)
    unitv2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unitv1, unitv2)
    angle = math.degrees(np.arccos(dot_product))
    if radian == True:
        angle = np.arccos(dot_product)

    return angle

#함수 모음

def distance(px, pz, ax, az):
    '''두 좌표사이 거리를 구함. x,z좌표 기준.'''
    distance = np.sqrt((px - ax)**2 + 
                           (pz - az)**2)
    return distance

def gettangentpoint(Px: float, Pz: float, Ax: float, Az: float, r: float, plot=False, title='tangent'):
    '''
    Get tangent point of a circle of center (Px,Pz) from a point outside the circle (Ax,Az)
    r is the radius of the circle.
    
    Px, Pz, Ax, Az, r = float or int
    
    Plotting is added
    https://stackoverflow.com/questions/49968720/find-tangent-points-in-a-circle-from-a-point
    '''
    #Get tangent point
    func = np.sqrt((Ax - Px)**2 + (Az - Pz)**2)  # player와 agent간 거리 구하기
    cosine = r / func # 밑변이 되는 r값(원의 반지름)과 빗변이 되는 player/agent 간 거리를 나누어 cosine 값을 구한다.
    cosine.loc[cosine>1] = 1 ## arccos는 -1,1 사이의 값만 받을 수 있는데, 계산상 오류인지 1을 넘는게 나와 clipping 해준다. 원리상 1을 넘을리 없으므로.
    ## -1의 경우, func 값과 r 값 모두 양수라 굳이 안해줬다.    
    th = np.arccos(cosine)  # angle theta. cosine 값에 역함수인 arccos를 취해서 세타 값을 얻어낸다. 이것이 agent가 player를 향한다고 한다면 취할 수 있는 최대 각도가 된다.
    direc = np.arctan2(Az - Pz, Ax - Px)  # direction angle of point A from P 
    ## 평면 좌표계에서 agent로 부터 player personal space 영역 원에 직교하는 두 점의 위치를 알기 위해, player위치로부터 수평인 선을 밑변으로 하는(빗변은 func)
    ## 직각 삼각형의 player 쪽 각도이다.
    d1 = direc + th  # direction angle of point T1 from Player
    d2 = direc - th  # direction angle of point T2 from Player
    ### 위 각도를 기반으로 평면 좌표계에서 두 접점의 좌표를 알아낼 수 있다.
    T1x = Px + r * np.cos(d1)
    T1z = Pz + r * np.sin(d1)
    T2x = Px + r * np.cos(d2)
    T2z = Pz + r * np.sin(d2)

    T1 = np.array([T1x, T1z])
    T2 = np.array([T2x, T2z])
    #For sanity check. Plotting
    if plot == True:
        #To plot Player circle
        x1=[] 
        z1=[]
        for theta in range(0,360):
            x1.append(Px + r*math.cos(math.radians(theta)))
            z1.append(Pz + r*math.sin(math.radians(theta)))
            
        plt.figure(figsize=(5,5))
        plt.grid()
        plt.title(title)
        #plot circle
        plt.plot(x1,z1)
        
        # plotting dots
        plt.plot(Px,Pz, marker='o', markersize=10) #Player
        plt.plot(Ax,Az, marker='o', markersize=10) #Agent
        plt.plot(T1x,T1z, marker='o', markersize=10) #First tangent point
        plt.plot(T2x,T2z, marker='o', markersize=10) #Secodn tangent point
        plt.axis('square')
        #Annotation
        player = 'P({},{})'.format(str(round(Px)),str(round(Pz)))
        plt.annotate(player, xy=(round(Px),round(Pz)))
        agent = 'A({},{})'.format(str(round(Ax)),str(round(Az)))
        plt.annotate(agent, xy=(round(Ax),round(Az)))
        tangent1 = 'T({},{})'.format(str(round(T1x)),str(round(T1z)))
        tangent2 = 'T({},{})'.format(str(round(T2x)),str(round(T2z)))
        plt.annotate(tangent1, xy=(round(T1x),round(T1z)))
        plt.annotate(tangent2, xy=(round(T2x),round(T2z)))

    return T1, T2

def to_dataframe(playertransform_list):
    '''Put time, subjectID, distance, visual degree into one dataframe'''
    df_whole_list = []
    agent_list = getvector(playertransform_list)
    frontvec_list = getfrontvec(playertransform_list)
    agent_list_degree = visualdegree(playertransform_list)
    distance = distance(playertransform_list)
    for i in range(len(distance)):
        df_processed = pd.DataFrame({'Time' : playertransform_list[i]['Time'], 
                             'subject_ID' : playertransform_list[i]['SubjectID_2_pos'],
                             'distance' : distance[i],
                             'visual_degree' : agent_list_degree[i]})
        #df_processed.set_index('Time', inplace=True, drop=True)
        
        df_whole_list.append(df_processed)
    
    return df_whole_list


def checkinvisual(playertransform_list, distance=2, visualdegree=55):
    '''Check whether the agent is currently in visual field or not'''
    df_whole = to_dataframe(playertransform_list)
    df_insight = pd.DataFrame({'Time' : playertransform_list[0]['Time']})
    df_insight['insight'] = 0
    
    

    for i in range(len(df_whole)):
        df_whole[i]['insight'] = 0
        for j in range(len(df_whole[i])):
            if df_whole[i]['distance'][j] <= distance and  df_whole[i]['visual_degree'][j] <= visualdegree:
                df_whole[i]['insight'][j] += 1
                df_insight['insight'][j] += 1

    return df_insight

def personalspace(playertransform_list, insight = True, visualdegree=55):
    '''Check whether the agent is currently in visual field or not'''
    df_whole = to_dataframe(playertransform_list)
    df_bubble = pd.DataFrame({'Time' : playertransform_list[0]['Time']})
    df_bubble['intimate space'] = 0
    df_bubble['personal space'] = 0
    df_bubble['social space'] = 0
    df_bubble['public space'] = 0
    intimate = 0.45
    personal = 1.2
    social = 3.5
    public = 7.6


    for i in range(len(df_whole)):
        for j in range(len(df_whole[i])):
            if insight == True:
                if df_whole[i]['distance'][j] <= intimate and  df_whole[i]['visual_degree'][j] <= visualdegree:
                    df_bubble['intimate space'][j] += 1
                elif df_whole[i]['distance'][j] <= personal and  df_whole[i]['visual_degree'][j] <= visualdegree:
                    df_bubble['personal space'][j] += 1
                elif df_whole[i]['distance'][j] <= social and  df_whole[i]['visual_degree'][j] <= visualdegree:
                    df_bubble['social space'][j] += 1
                elif df_whole[i]['distance'][j] <= public and  df_whole[i]['visual_degree'][j] <= visualdegree:
                    df_bubble['public space'][j] += 1
            if insight == False:
                if df_whole[i]['distance'][j] <= intimate:
                    df_bubble['intimate space'][j] += 1
                elif df_whole[i]['distance'][j] <= personal:
                    df_bubble['personal space'][j] += 1
                elif df_whole[i]['distance'][j] <= social:
                    df_bubble['social space'][j] += 1
                elif df_whole[i]['distance'][j] <= public:
                    df_bubble['public space'][j] += 1

    #df_insight = df_insight.set_index('Time', drop=True)
    return df_bubble

def closestagentdistance(playertransform_list, insight = True, visualdegree=55):
    '''Get distance between player and closest agent. if insight = True, only insight agent is checked'''
    df_whole_list = to_dataframe(playertransform_list)
    df_closestdist = pd.DataFrame({'Time' : playertransform_list[0]['Time']})
    df_closestdist['closest_distance'] = 0
    df_distance_out =  pd.DataFrame({'Time' :playertransform_list[0]['Time']})
    if insight == True:
        # Insight + True
        df_insightonly = df_whole_list.copy()
        for agent in range(len(df_insightonly)):
            for index in range(len(df_insightonly[agent])):
                if df_insightonly[agent]['visual_degree'][index] > visualdegree:
                    df_insightonly[agent]['distance'][index] = 99999999
        for index in range(len(df_insightonly)):
            distance_col = df_insightonly[index]['distance']
            name = '_' + str(df_insightonly[index]['subject_ID'][0])
            df_distance_out = df_distance_out.join(distance_col,rsuffix = name)

    
    else:
        # Insight = False
        for index in range(len(df_whole_list)):
            distance_col = df_whole_list[index]['distance']
            name = '_' + str(df_whole_list[index]['subject_ID'][0])
            df_distance_out = df_distance_out.join(distance_col,rsuffix = name)

    first_name = 'distance_' + str(df_whole_list[0]['subject_ID'][0])
    df_distance_out.rename(columns = {'distance':first_name}, inplace = True)
    df_distance_out = df_distance_out.iloc[ : , :-1]

    df_closestdist['closest_distance'] = df_distance_out.min(axis = 1)

    return df_closestdist
    

def insightsec(playertransform_list, distance = 2):
    '''Get total time that agents was in sight of the player'''
    df_insight = checkinvisual(playertransform_list, distance)
    df_insight_1 = df_insight[df_insight['insight']== 1] 
    df_insight_1 = df_insight_1.reset_index().drop('index',axis=1)

    insight_sec_list = []
    startpoint = 0  #start point of a single event.
    insight_sec = 0
    for index in  range(len(df_insight_1)):
        if index < len(df_insight_1)-1:
            if df_insight_1['Time'][index+1] - df_insight_1['Time'][index] > 1:  
                #if the time gap between rows are bigger than 1, count it as a new event.           
                insight_sec = df_insight_1['Time'][index] - df_insight_1['Time'][startpoint]
                insight_sec_list.append(insight_sec)
                startpoint = index + 1

        else:
            insight_sec =  df_insight_1['Time'][index] - df_insight_1['Time'][startpoint]
            insight_sec_list.append(insight_sec)
    
    insightsec = sum(insight_sec_list)
    return insightsec

def anglebetween(v1, v2):
    '''
    Get angle between two angles
    '''
    unit_vector_1 = v1.T / np.linalg.norm(v1, axis=1)  ## u/|u|
    unit_vector_2 = v2.T / np.linalg.norm(v2, axis=1)  ## v/|v|
    cos = np.einsum("ij,ij->i", unit_vector_1.T, unit_vector_2.T) ## u*v/|u|*|v| = cos() by cosine제 2법칙
    
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    
    return angle

def agentcheck(subj_p, subj_r, names, zone_f = 7.6, zone_c = 1.2, visual_degree = 50):
    '''
    This is function to organize player and agent relationship.
    Result will contain player-agent distance, player-agent degree(to check whether the agent is inside visual degree)
    and intention of agent, calculated by agent movement vector and player visual degree, within certain distance(zonef and c)
    
    subj_p: subject position dataframe list
    subj_r: subject rotation dataframe list
    names: Player name list
    zone_f: Zone Far. Zone that counted as 'in visual distance'. Public Zone distance is used
    zone_c: Zone Close. Zone that Participant will think as their personal zone
    visual_degree: Visual degree is reduced to 100 from 110 to make sure agent is in the visual field considering Sensor issue and attention issue
    
    '''
    totals = []

    for name in range(len(names)):
        total = []
        # Player and Agent position data
        pt = playertransform(subj_p[name].drop(["Subject"], axis=1), subj_r[name].drop(["Subject"], axis=1))
        #Distance and Degree check between Agent and Player
        
        for a in range(len(pt)): #Remove player df from pt
            if pt[a]['SubjectID_1_pos'][0] == pt[a]['SubjectID_2_pos'][0]:
                del pt[a]
        
        dist = distance(pt)
        deg = visualdegree(pt)

        for a in range(len(pt)):
            intentions = [] #Contain each intention label of agent by time
            for idx in range(len(pt[a])):
                intention = 'unrelated'
                
                dist_c = dist[a][idx]
                deg_c = deg[a][idx]
                if dist_c <= zone_f and deg_c < visual_degree:
                    if dist_c > zone_c:
                        # Current df of agent
                        current = pt[a]
                        # Player position
                        Px = current["X_1_pos"][idx]
                        Pz = current["Z_1_pos"][idx]
                        Ax = current["X_2_pos"][idx]
                        Az = current["Z_2_pos"][idx]
                        r = zone_c
                        # print(r/sqrt((Ax - Px)**2 + (Az - Pz)**2))
                        T, _ = gettangentpoint(Px,Pz,Ax,Az,r)
                        
                        #Agent Front Vector
                        Afx = 5*math.cos(math.pi * (Ax / 180)) #degree to radian. 5 is arbitrary lenth of the vector
                        Afz = 5*math.sin(math.pi * (Ax / 180))
                        # Vectors
                        ##Agent to Player Vector
                        AP = [Px-Ax, Pz-Az]
                        #Tangent Vector
                        V1 =  [T[0]-Ax, T[1]-Az]
                        #Agent Front Vector(Direction Vector)
                        AD = [Afx, Afz]
                        #Degree Processing
                        AP_T_ang = anglebetween(AP, V1) # Tangent Degree
                        AP_AD_ang = anglebetween(AP, AD) # Agend Direction Vector degree
                        #Check Whether Agent is directed to Player or Not
                        if AP_T_ang >= AP_AD_ang: #Directed to player
                            intention = 'direct toward'
                        
                        if AP_T_ang < AP_AD_ang: #Moving away or unrelevant to player
                            intention = 'unrelated'
                        
                    if dist_c <= zone_c:
                        intention = 'inside zone'
                        
                intentions.append(intention)
            
            # create total df
            df = pd.DataFrame({'Time' : pt[a]['Time'], 
                'subject_ID' : pt[a]['SubjectID_2_pos'],
                'distance' : dist[a],
                'visual_degree' : deg[a],
                'intention': intentions})
            total.append(df)
            
        totals.append(total)
    
    return totals
    
    
import os
import traceback
from glob import glob
import pandas as pd
from collections import defaultdict
import numpy as np
from Dataloader import (
    read_acqknowledge_with_markers,
    xdfreaderfixer,
    rename_duplicates,
    read_csv_with_max_columns,
    Anxietyloader
)
from neurokit2.misc import as_vector

def process_physiology_signals(result, sampling_rate=2000, target_rate=120):
    import neurokit2 as nk
    import pandas as pd
    import numpy as np
    from neurokit2.misc import as_vector

    # ------------------ Step 1. 계산 준비 ------------------
    orig_len = len(result)
    target_len = int(orig_len * target_rate / sampling_rate)
    index = (np.linspace(0, orig_len - 1, num=target_len)).astype(int)

    # ------------------ Step 2. Signal resample ------------------
    signals = {}
    for sig in ["PPG", "EDA", "RSP"]:
        signals[sig] = nk.signal_resample(
            result[sig].values,
            sampling_rate=sampling_rate,
            desired_sampling_rate=target_rate
        )
    signals_df = pd.DataFrame(signals)

    # ------------------ Step 3. Metadata downsample (index-based) ------------------
    scene_ds = result["scene"].iloc[index].reset_index(drop=True)
    marker_ds = result["marker"].iloc[index].reset_index(drop=True)

    # ------------------ Step 4. Preprocessing on 120Hz ------------------

    ## --- PPG ---
    methods_ppg = nk.ppg_methods(sampling_rate=target_rate, method="elgendi", method_quality="zhao2018")
    # methods_ppg["kwargs_quality"]["approach"] = None
    ppg_clean = nk.ppg_clean(signals_df["PPG"], sampling_rate=target_rate,
                             method=methods_ppg["method_cleaning"],
                             **methods_ppg["kwargs_cleaning"])
    peaks_ppg, info_ppg = nk.ppg_peaks(ppg_clean, sampling_rate=target_rate,
                                       method=methods_ppg["method_peaks"],
                                       correct_artifacts=True,
                                       **methods_ppg["kwargs_peaks"])
    ppg_rate = nk.signal_rate(info_ppg["PPG_Peaks"], sampling_rate=target_rate, desired_length=len(ppg_clean))
    # ppg_quality = nk.ppg_quality(ppg_clean, info_ppg["PPG_Peaks"], sampling_rate=target_rate,
    #                              method="templatematch", approach=None)
    df_ppg = pd.DataFrame({
        "PPG_Raw": signals_df["PPG"],
        "PPG_Clean": ppg_clean,
        "PPG_Rate": ppg_rate,
        # "PPG_Quality": ppg_quality,
        "PPG_Peaks": peaks_ppg["PPG_Peaks"].values
    })

    ## --- EDA ---
    eda_clean = nk.eda_clean(signals_df["EDA"], sampling_rate=target_rate, method="biosppy")
    eda_decomp = nk.eda_phasic(eda_clean, sampling_rate=target_rate, method="highpass", cutoff=0.05)
    peaks_eda, info_eda = nk.eda_peaks(eda_decomp["EDA_Phasic"], sampling_rate=target_rate,
                                       method="neurokit", amplitude_min=0.1)
    df_eda = pd.DataFrame({
        "EDA_Raw": signals_df["EDA"],
        "EDA_Clean": eda_clean
    }).join([eda_decomp, peaks_eda])

    ## --- RSP ---
    rsp_clean = nk.rsp_clean(signals_df["RSP"], sampling_rate=target_rate)
    peaks_rsp, info_rsp = nk.rsp_peaks(rsp_clean, sampling_rate=target_rate)
    rsp_phase = nk.rsp_phase(peaks_rsp, desired_length=len(rsp_clean))
    rsp_amp = nk.rsp_amplitude(rsp_clean, peaks_rsp)
    rsp_rate = nk.signal_rate(info_rsp["RSP_Troughs"], sampling_rate=target_rate, desired_length=len(rsp_clean))
    rsp_sym = nk.rsp_symmetry(rsp_clean, peaks_rsp)
    rsp_rvt = nk.rsp_rvt(rsp_clean, method="harrison2021", sampling_rate=target_rate, silent=True)
    df_rsp = pd.DataFrame({
        "RSP_Raw": signals_df["RSP"],
        "RSP_Clean": rsp_clean,
        "RSP_Amplitude": rsp_amp,
        "RSP_Rate": rsp_rate,
        "RSP_RVT": rsp_rvt
    }).join([rsp_phase, rsp_sym, peaks_rsp])

    # ------------------ Step 5. Merge all ------------------
    physiology = pd.concat([df_ppg, df_eda, df_rsp], axis=1)
    physiology["scene"] = scene_ds
    physiology["marker"] = marker_ds

    return physiology, df_ppg, df_eda, df_rsp, info_ppg, info_eda, info_rsp

### --- 1. Load Single Participant --- ###
def dataloader_one(participant_path: str, scenes: list, preloaded_result_df=None):
    import os
    import traceback
    from glob import glob
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from Dataloader import (
        read_acqknowledge_with_markers,
        xdfreaderfixer,
        rename_duplicates,
        read_csv_with_max_columns,
        Anxietyloader,
        process_physiology_signals,
        sync_biopac_markers_with_unity,
    )

    imputed_markers_df = pd.DataFrame()  # 또는 None
    name = os.path.basename(participant_path)
    print(f"🔹 Loading participant: {name}")

    data = {
        "Psychopy": [],
        "Physiology": [],
        "Tracker": [],
        "Eyetracker": [],
        "Facetracker": []
    }
    unity = defaultdict(dict)
    warnings = []

    try:
        # -------------------------------------------------
        # 1. Load anxiety (Psychopy log)
        # -------------------------------------------------
        anxiety_logs = glob(os.path.join(participant_path, '*anxiety.log'))
        if anxiety_logs:
            for log in anxiety_logs:
                try:
                    anxiety_df = Anxietyloader(log, name, scenes.copy())
                    data["Psychopy"].append(anxiety_df)
                except Exception as e:
                    print(f"⚠️ Failed to load anxiety log for {name}: {e}")
                    warnings.append("Psychopy")
        else:
            print(f"⚠️ No anxiety log found for {name}")
            warnings.append("Psychopy")

        # -------------------------------------------------
        # 2. Load xdf and CSV (Unity-related CSV 포함)
        # -------------------------------------------------
        for file in os.listdir(participant_path):
            file_path = os.path.join(participant_path, file)
            file_ext = file.split(".")[-1]

            # 2-1) Tracker (xdf)
            if file_ext == "xdf":
                try:
                    tracker_data = xdfreaderfixer(file_path)
                    data["Tracker"].append(tracker_data)
                except Exception as e:
                    print(f"⚠️ Failed to load xdf for {name}: {e}")
                    warnings.append("Tracker")

            # 2-2) Unity CSV (Customevent / Agent / SUDS 등)
            elif file_ext == "csv":
                # 파일명 예: 'Elevator1_Customevent.csv'
                scene = file.split("_")[0]
                dtype_raw = file.split("_")[1].split(".")[0]  # 예: 'Customevent'
                dtype = dtype_raw  # 원래 키
                dtype_lower = dtype_raw.lower()

                # SUDS는 여기서 제외 (이미 Psychopy에서 다룸)
                if dtype_lower != "suds":
                    try:
                        df = read_csv_with_max_columns(file_path)
                        df.columns = df.columns.str.strip()
                        df['scene'] = scene
                        df['Subject'] = name

                        # 기존 키와 소문자 키 둘 다에 저장 → 호환성 유지
                        unity[dtype][scene] = df
                        unity[dtype_lower][scene] = df
                    except Exception as e:
                        print(f"⚠️ Failed to load CSV {file} for {name}: {e}")
                        warnings.append(f"CSV:{file}")

        # -------------------------------------------------
        # 3. Split Tracker → Eyetracker / Facetracker
        # -------------------------------------------------
        if data["Tracker"]:
            try:
                # xdfreaderfixer 반환 형식이 [df, info]라고 가정
                df = data["Tracker"][0][0]
                df.columns = rename_duplicates([col.strip() for col in df.columns])

                mandatory = ['Scene', 'UnityTime']
                unit_cols = [col for col in df.columns if 'unit' in col]

                face = df[mandatory + unit_cols]
                eye = df[mandatory + [col for col in df.columns if col not in mandatory + unit_cols]]

                mapping = {
                    0: "Start",
                    1: "Practice",
                    2: "ElevatorTest",
                    3: "Elevator1",
                    4: "Outside",
                    5: "Hallway",
                    6: "Elevator2",
                    7: "Hall",
                    8: "End"
                }

                eye.loc[:, 'Scene'] = eye['Scene'].map(mapping)
                face.loc[:, 'Scene'] = face['Scene'].map(mapping)
                eye = eye.rename(columns={"Scene": "scene"})
                face = face.rename(columns={"Scene": "scene"})
                eye['Subject'] = name
                face['Subject'] = name

                data["Eyetracker"].append(eye)
                data["Facetracker"].append(face)
            except Exception as e:
                print(f"⚠️ Failed to split Tracker for {name}: {e}")
                warnings.extend(["Eyetracker", "Facetracker"])

        # -------------------------------------------------
        # 4. Load Physiology (.acq) + Unity 기반 StartP/End sync
        #     + (옵션) 필터된 RSP 덮어쓰기
        # -------------------------------------------------
        # 4-1) VR .acq 파일 찾기
        vr_file_path = None
        for file in os.listdir(participant_path):
            file_ext = file.split(".")[-1]
            if file_ext == "acq" and "VR" in file and "BT" not in file:
                vr_file_path = os.path.join(participant_path, file)
                break

        if vr_file_path is None:
            raise FileNotFoundError(f"No VR .acq file found in {participant_path}")

        # 4-2) 원본 VR.acq → Biopac 신호 + event_markers 로딩
        try:
            result_raw, event_markers_df, fs = read_acqknowledge_with_markers(vr_file_path)
        except Exception as e:
            print(f"⚠️ Failed to read VR .acq for {name}: {e}")
            warnings.append("Physiology")
            raise

        # 컬럼 정리 및 채널 rename
        result_raw.columns = result_raw.columns.str.strip()
        result_raw.rename(columns={
            "EDA, Y, PPGED-R": "EDA",
            "RSP, X, RSPEC-R": "RSP",
            "PPG, X, PPGED-R": "PPG",
        }, inplace=True)
        result_raw["Subject"] = name

        # 4-3) Unity Customevent 기반 StartP/End 보정 → scene / marker 재할당
        try:
            result_synced, imputed_markers_df = sync_biopac_markers_with_unity(
                df_raw=result_raw,
                event_markers_df=event_markers_df,
                unity_dict=unity,
                scenes=scenes,
                fs=fs if fs is not None else 2000,
                unity_hz=120,
            )
        except Exception as e:
            print(f"⚠️ sync_biopac_markers_with_unity failed for {name}: {e}")
            # 실패하면 원본 marker/scene 사용 (최소한의 fallback)
            result_synced = result_raw.copy()
            if "marker" not in result_synced.columns:
                result_synced["marker"] = "Ongoing"
            if "scene" not in result_synced.columns:
                result_synced["scene"] = "Unknown"
            warnings.append("MarkerSync")
            imputed_markers_df = pd.DataFrame()  # 또는 None

        # 4-4) (옵션) 필터된 RSP 덮어쓰기
        try:
            if preloaded_result_df is not None:
                filt = preloaded_result_df.copy()
                filt.columns = filt.columns.str.strip()

                # Sample 기준 merge가 가능한 경우
                if "Sample" in filt.columns and "Sample" in result_synced.columns:
                    filt_rsp = filt[["Sample", "RSP"]].copy()
                    result_synced = pd.merge(
                        result_synced.drop(columns=["RSP"], errors="ignore"),
                        filt_rsp,
                        on="Sample",
                        how="left",
                        validate="one_to_one",
                    )
                else:
                    # Sample 없으면 길이 기준으로 align 시도
                    if len(filt) != len(result_synced):
                        print(
                            f"⚠️ preloaded RSP length ({len(filt)}) "
                            f"!= raw VR length ({len(result_synced)}) for {name}; using raw RSP instead."
                        )
                    else:
                        result_synced["RSP"] = filt["RSP"].to_numpy()
        except Exception as e:
            print(f"⚠️ Failed to merge filtered RSP for {name}: {e}")
            warnings.append("FilteredRSPMerge")
        # 4-5) 전체 synced 구간에서 physiology 추출 후,
        #      RSP_Phase NaN 프레임은 버리고, 그 다음에 Ongoing만 사용
        try:
            if "marker" not in result_synced.columns:
                result_synced["marker"] = "Ongoing"
            if "scene" not in result_synced.columns:
                result_synced["scene"] = "Unknown"

            # 🔹 (1) ElevatorTest 포함 전체 구간에 대해 physiology 계산
            physiology_full, *_ = process_physiology_signals(
                result_synced,
                sampling_rate=fs if fs is not None else 2000,
                target_rate=120,
            )

            # 🔹 (3) 그 중에서 marker == "Ongoing" 인 부분만 사용
            physiology = physiology_full[physiology_full["marker"] == "Ongoing"].reset_index(drop=True)

            data["Physiology"].append(physiology)

        except Exception as e:
            print(f"⚠️ Failed to process physiology for {name}: {e}")
            warnings.append("Physiology")

        # # 4-5) marker == 'Ongoing' 구간만 사용 → process_physiology_signals
        # try:
        #     if "marker" not in result_synced.columns:
        #         result_synced["marker"] = "Ongoing"
        #     if "scene" not in result_synced.columns:
        #         result_synced["scene"] = "Unknown"

        #     result_ongoing = result_synced[result_synced["marker"] == "Ongoing"]

        #     physiology, *_ = process_physiology_signals(
        #         result_ongoing,
        #         sampling_rate=fs if fs is not None else 2000,
        #         target_rate=120,
        #     )
        #     data["Physiology"].append(physiology)
        # except Exception as e:
        #     print(f"⚠️ Failed to process physiology for {name}: {e}")
        #     warnings.append("Physiology")

    except Exception as e:
        print(f"🚨 Critical error loading participant {name}: {e}")
        traceback.print_exc()
        warnings.append("CriticalError")

    return name, data, unity, list(set(warnings)), imputed_markers_df

# def dataloader_one(participant_path: str, scenes: list, preloaded_result_df=None):
#     import os
#     import traceback
#     from glob import glob
#     from collections import defaultdict
#     import pandas as pd
#     import numpy as np
#     from Dataloader import (
#         read_acqknowledge_with_markers,
#         xdfreaderfixer,
#         rename_duplicates,
#         read_csv_with_max_columns,
#         Anxietyloader,

#     )

#     name = os.path.basename(participant_path)
#     print(f"🔹 Loading participant: {name}")

#     data = {
#         "Psychopy": [],
#         "Physiology": [],
#         "Tracker": [],
#         "Eyetracker": [],
#         "Facetracker": []
#     }
#     unity = defaultdict(dict)
#     warnings = []

#     try:
#         # 1. Load anxiety
#         anxiety_logs = glob(os.path.join(participant_path, '*anxiety.log'))
#         if anxiety_logs:
#             for log in anxiety_logs:
#                 try:
#                     anxiety_df = Anxietyloader(log, name, scenes.copy())
#                     data["Psychopy"].append(anxiety_df)
#                 except Exception as e:
#                     print(f"⚠️ Failed to load anxiety log for {name}: {e}")
#                     warnings.append("Psychopy")
#         else:
#             print(f"⚠️ No anxiety log found for {name}")
#             warnings.append("Psychopy")
#         # 2. Load Physiology (.acq)
#         if preloaded_result_df is not None:
#             try:
#                 # 2-1) 원본 VR.acq 파일 경로 찾기
#                 vr_file_path = None
#                 for file in os.listdir(participant_path):
#                     file_ext = file.split(".")[-1]
#                     if file_ext == "acq" and "VR" in file and "BT" not in file:
#                         vr_file_path = os.path.join(participant_path, file)
#                         break
#                 if vr_file_path is None:
#                     raise FileNotFoundError(f"No VR .acq file found in {participant_path}")

#                 # 2-2) 원본 VR.acq에서 physiology + marker/scene 로딩
#                 result, _, _ = read_acqknowledge_with_markers(vr_file_path)
#                 result.columns = result.columns.str.strip()
#                 result.rename(columns={
#                     "EDA, Y, PPGED-R": "EDA",
#                     "RSP, X, RSPEC-R": "RSP",
#                     "PPG, X, PPGED-R": "PPG",
#                 }, inplace=True)
#                 result["Subject"] = name
#                 result["marker"] = result.get("marker", "Ongoing")
#                 result["scene"] = result.get("scene", "Unknown")

#                 # 2-3) 필터된 RSP와 merge
#                 filt = preloaded_result_df.copy()
#                 filt.columns = filt.columns.str.strip()

#                 if "Sample" in filt.columns and "Sample" in result.columns:
#                     # Sample 기준으로 one-to-one merge
#                     filt_rsp = filt[["Sample", "RSP"]].copy()
#                     result = pd.merge(
#                         result.drop(columns=["RSP"], errors="ignore"),
#                         filt_rsp,
#                         on="Sample",
#                         how="left",
#                         validate="one_to_one",
#                     )
#                 else:
#                     # Sample이 없으면 길이 기반 align (fallback)
#                     if len(filt) != len(result):
#                         print(
#                             f"⚠️ preloaded RSP length ({len(filt)}) "
#                             f"!= raw VR length ({len(result)}) for {name}; using raw RSP instead."
#                         )
#                     else:
#                         result["RSP"] = filt["RSP"].to_numpy()

#                 # 2-4) marker가 Ongoing인 구간만 사용
#                 result = result[result["marker"] == "Ongoing"]

#                 physiology, *_ = process_physiology_signals(result, sampling_rate=2000)
#                 data["Physiology"].append(physiology)
#             except Exception as e:
#                 print(f"⚠️ Failed to use preloaded physiology df for {name}: {e}")
#                 warnings.append("Physiology")
#         else:
#             for file in os.listdir(participant_path):
#                 file_path = os.path.join(participant_path, file)
#                 file_ext = file.split(".")[-1]

#                 if file_ext == "acq" and "VR" in file and "BT" not in file:
#                     try:
#                         result, _, _ = read_acqknowledge_with_markers(file_path)
#                         result.columns = result.columns.str.strip()
#                         result.rename(columns={
#                             "EDA, Y, PPGED-R": "EDA",
#                             "RSP, X, RSPEC-R": "RSP",
#                             "PPG, X, PPGED-R": "PPG",
#                         }, inplace=True)
#                         result["Subject"] = name
#                         result["marker"] = result.get("marker", "Ongoing")
#                         result["scene"] = result.get("scene", "Unknown")
#                         result = result[result["marker"] == "Ongoing"]

#                         physiology, *_ = process_physiology_signals(result, sampling_rate=2000)
#                         data["Physiology"].append(physiology)
#                     except Exception as e:
#                         print(f"⚠️ Failed to process .acq for {name}: {e}")
#                         warnings.append("Physiology")

#         # # 2. Load Physiology (.acq)
#         # if preloaded_result_df is not None:
#         #     try:
#         #         result = preloaded_result_df.copy()
#         #         result.columns = result.columns.str.strip()
#         #         result["Subject"] = name
#         #         result["marker"] = result.get("marker", "Ongoing")
#         #         result["scene"] = result.get("scene", "Unknown")
#         #         result = result[result["marker"] == "Ongoing"]

#         #         # ✅ 이 시점에서 result는 이미 RSP가 필터링된 값
#         #         physiology, *_ = process_physiology_signals(result, sampling_rate=2000)
#         #         data["Physiology"].append(physiology)
#         #     except Exception as e:
#         #         print(f"⚠️ Failed to use preloaded physiology df for {name}: {e}")
#         #         warnings.append("Physiology")
#         # else:
#         #     for file in os.listdir(participant_path):
#         #         file_path = os.path.join(participant_path, file)
#         #         file_ext = file.split(".")[-1]

#         #         if file_ext == "acq" and "VR" in file and "BT" not in file:
#         #             try:
#         #                 result, _, _ = read_acqknowledge_with_markers(file_path)
#         #                 result.columns = result.columns.str.strip()
#         #                 result.rename(columns={
#         #                     "EDA, Y, PPGED-R": "EDA",
#         #                     "RSP, X, RSPEC-R": "RSP",
#         #                     "PPG, X, PPGED-R": "PPG",
#         #                 }, inplace=True)
#         #                 result["Subject"] = name
#         #                 result["marker"] = result.get("marker", "Ongoing")
#         #                 result["scene"] = result.get("scene", "Unknown")
#         #                 result = result[result["marker"] == "Ongoing"]

#         #                 physiology, *_ = process_physiology_signals(result, sampling_rate=2000)
#         #                 data["Physiology"].append(physiology)
#         #             except Exception as e:
#         #                 print(f"⚠️ Failed to process .acq for {name}: {e}")
#         #                 warnings.append("Physiology")

#         # 3. Load xdf and csv
#         for file in os.listdir(participant_path):
#             file_path = os.path.join(participant_path, file)
#             file_ext = file.split(".")[-1]

#             if file_ext == "xdf":
#                 try:
#                     tracker_data = xdfreaderfixer(file_path)
#                     data["Tracker"].append(tracker_data)
#                 except Exception as e:
#                     print(f"⚠️ Failed to load xdf for {name}: {e}")
#                     warnings.append("Tracker")

#             elif file_ext == "csv":
#                 scene = file.split("_")[0]
#                 dtype = file.split("_")[1].split(".")[0]
#                 if dtype != "SUDS":
#                     try:
#                         df = read_csv_with_max_columns(file_path)
#                         df.columns = df.columns.str.strip()
#                         df['scene'] = scene
#                         df['Subject'] = name
#                         unity[dtype][scene] = df
#                     except Exception as e:
#                         print(f"⚠️ Failed to load CSV {file} for {name}: {e}")
#                         warnings.append(f"CSV:{file}")

#         # 4. Split tracker if loaded
#         if data["Tracker"]:
#             try:
#                 df = data["Tracker"][0][0]
#                 df.columns = rename_duplicates([col.strip() for col in df.columns])
#                 mandatory = ['Scene', 'UnityTime']
#                 unit_cols = [col for col in df.columns if 'unit' in col]
#                 face = df[mandatory + unit_cols]
#                 eye = df[mandatory + [col for col in df.columns if col not in mandatory + unit_cols]]

#                 mapping = {0: "Start", 1: "Practice", 2: "ElevatorTest", 3: "Elevator1",
#                            4: "Outside", 5: "Hallway", 6: "Elevator2", 7: "Hall", 8: "End"}

#                 eye.loc[:, 'Scene'] = eye['Scene'].map(mapping)
#                 face.loc[:, 'Scene'] = face['Scene'].map(mapping)
#                 eye = eye.rename(columns={"Scene": "scene"})
#                 face = face.rename(columns={"Scene": "scene"})
#                 eye['Subject'] = name
#                 face['Subject'] = name

#                 data["Eyetracker"].append(eye)
#                 data["Facetracker"].append(face)
#             except Exception as e:
#                 print(f"⚠️ Failed to split Tracker for {name}: {e}")
#                 warnings.extend(["Eyetracker", "Facetracker"])

#     except Exception as e:
#         print(f"🚨 Critical error loading participant {name}: {e}")
#         traceback.print_exc()
#         warnings.append("CriticalError")

#     return name, data, unity, list(set(warnings))


# Unity
def clean_unity(df, dtype):
    df.columns = df.columns.str.strip()
    if dtype == "customevent":
        if "Actor" in df.columns:
            df["SubjectID"] = df["Actor"].str.replace('"', '', regex=False)
            df.drop(columns=["Actor"], inplace=True)
        df.columns = [f"customevent_col_{i}" if col == '' else col for i, col in enumerate(df.columns)]

        for col in ["Action", "1", "2", "3", "4"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[()"]', '', regex=True).str.strip()
    return df
# Strip + Filter
def strip_and_filter(df, scenes):
    df.columns = df.columns.str.strip()
    return df[df["scene"].isin(scenes)]
### --- 2. Preprocess and Merge --- ###
### ✅ 수정된 `preprocess_and_merge` 및 새로운 `extract_agent_df` 함수 ###
import pandas as pd
import numpy as np

def preprocess_and_merge(data: dict, unity: dict, scenes: list, target_hz=120, physio_hz=2000):
    warnings = []

    # --- 1. Load each source safely ---
    def safe_get(key):
        if not data.get(key) or len(data[key]) == 0:
            warnings.append(f"{key}_missing")
            return pd.DataFrame()
        return data[key][0].copy()

    physiology_df = safe_get('Physiology')
    eye = safe_get('Eyetracker')
    face = safe_get('Facetracker')
    anxiety_df = safe_get('Psychopy')

    # --- 2. HeadCollider ID 찾기 ---
    head_ids = []
    for scene in scenes:
        if scene in unity.get('subjects', {}):
            subj_df = unity['subjects'][scene].copy()
            subj_df.columns = subj_df.columns.str.strip()
            subj_df['Name'] = subj_df['Name'].astype(str).str.strip()
            try:
                head_id = subj_df.loc[subj_df['Name'] == 'HeadCollider', 'ID'].values[0]
                head_ids.append(head_id)
            except IndexError:
                warnings.append(f"HeadCollider_missing_in_{scene}")

    use_position_rotation = bool(head_ids)
    if not use_position_rotation:
        print("🚨 No HeadCollider found → skipping position/rotation merge.")

    # --- 3. Unity position/rotation 로딩 ---
    def load_unity_data(dtype):
        return pd.concat([unity[dtype][s] for s in scenes if s in unity.get(dtype, {})], ignore_index=True) \
            if dtype in unity else pd.DataFrame()

    unity_position_df = clean_unity(load_unity_data('position'), "position")
    unity_rotation_df = clean_unity(load_unity_data('rotation'), "rotation")

    # --- 4. Strip + Filter ---
    def clean_and_filter(df):
        df.columns = df.columns.str.strip()
        return df[df["scene"].isin(scenes)] if not df.empty and "scene" in df.columns else df

    physiology_df = clean_and_filter(physiology_df)
    eye = clean_and_filter(eye)
    face = clean_and_filter(face)
    anxiety_df = clean_and_filter(anxiety_df)

                
    if use_position_rotation:
        unity_position_df = clean_and_filter(unity_position_df)
        unity_rotation_df = clean_and_filter(unity_rotation_df)

        unity_position_df = unity_position_df[unity_position_df["SubjectID"].isin(head_ids)].copy()
        unity_rotation_df = unity_rotation_df[unity_rotation_df["SubjectID"].isin(head_ids)].copy()

    # --- 5. Frame conversion ---
    for df, col in [(eye, "UnityTime"), (face, "UnityTime")]:
        if not df.empty:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=[col], inplace=True)
            df["time"] = (df[col] * target_hz).astype(int)

    if use_position_rotation:
        for df in [unity_position_df, unity_rotation_df]:
            if not df.empty:
                df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
                df.dropna(subset=["Time"], inplace=True)
                df["time"] = (df["Time"] * target_hz).astype(int)

    # --- 6. Downsample physiology ---
    if not physiology_df.empty:
        # factor = physio_hz // target_hz
        # physiology_df = physiology_df.iloc[::factor].reset_index(drop=True)
        physiology_df["time"] = np.arange(len(physiology_df))

    # --- 7. Resample (interpolate missing frames) ---
    def resample(df):
        if df.empty: return df
        df = df.drop_duplicates(subset=["time"])
        df = df.set_index("time")
        df = df.reindex(range(df.index.min(), df.index.max() + 1))
        return df.ffill().reset_index().rename(columns={"index": "time"})

    eye = resample(eye)
    face = resample(face)
    physiology_df = resample(physiology_df)
    if use_position_rotation:
        unity_position_df = resample(unity_position_df)
        unity_rotation_df = resample(unity_rotation_df)

    # --- 8. Normalize time per scene ---
    for df in [eye, face, physiology_df]:
        if df.empty: continue
        for scene in scenes:
            mask = df["scene"] == scene
            if mask.any():
                df.loc[mask, "time"] -= df.loc[mask, "time"].min()

    if use_position_rotation:
        for df in [unity_position_df, unity_rotation_df]:
            if df.empty: continue
            for scene in scenes:
                mask = df["scene"] == scene
                if mask.any():
                    df.loc[mask, "time"] -= df.loc[mask, "time"].min()
    
    # ✅ 내부 컬럼 제거 추가
    
    for df in [eye, face, physiology_df]:
        for col in ["UnityTime", "SubjectID", "convergence_distance_mm", "convergence_distance_validity"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    # --- 9. Merge step by step ---
    merged = pd.merge_asof(eye.sort_values("time"), face.sort_values("time"), on="time", by="scene", direction="nearest")
    if use_position_rotation and not unity_position_df.empty:
        merged = pd.merge_asof(merged.sort_values("time"), unity_position_df.sort_values("time"), on="time", by="scene", direction="nearest")
        merged = pd.merge_asof(merged.sort_values("time"), unity_rotation_df.sort_values("time"), on="time", by="scene", direction="nearest")
    if not physiology_df.empty:
        merged = pd.merge_asof(merged.sort_values("time"), physiology_df.sort_values("time"), on="time", by="scene", direction="nearest")

    # --- 10. Merge anxiety ---
    if not anxiety_df.empty:
        anxiety_df["time"] -= anxiety_df["time"].min()
        anxiety_df["time"] = (anxiety_df["time"] * target_hz).astype(int)

        def smart_fill(group):
            first_valid = group["slider marker"].first_valid_index()
            if first_valid is not None:
                group.loc[:first_valid, "slider marker"] = group.loc[:first_valid, "slider marker"].bfill()
                group["slider marker"] = group["slider marker"].ffill()
            return group

        anxiety_df = anxiety_df.sort_values(["scene", "time"]).reset_index(drop=True)
        anxiety_df = anxiety_df.groupby("scene", group_keys=False).apply(smart_fill)

        def reindex_scene(group):
            group = group.drop_duplicates(subset=["time"]).set_index("time")
            full_range = range(group.index.min(), group.index.max() + 1)
            group = group.reindex(full_range)
            group["slider marker"] = group["slider marker"].ffill()
            group["scene"] = group["scene"].ffill()
            group["name"] = group["name"].ffill()
            return group.reset_index().rename(columns={"index": "time"}).assign(time=lambda df: df["time"] - df["time"].min())

        anxiety_df = anxiety_df.groupby("scene", group_keys=False).apply(reindex_scene)

        merged_list = []
        for scene in scenes:
            base = merged[merged["scene"] == scene].sort_values("time")
            overlay = anxiety_df[anxiety_df["scene"] == scene].sort_values("time")
            if base.empty or overlay.empty:
                continue
            merged_scene = pd.merge_asof(base, overlay, on="time", direction="nearest")
            merged_scene["scene"] = scene
            merged_list.append(merged_scene)

        if merged_list:
            merged = pd.concat(merged_list, ignore_index=True)
        else:
            warnings.append("Anxiety_merge_skipped")
            merged["anxiety"] = np.nan
    else:
        merged["anxiety"] = np.nan
        warnings.append("Anxiety_missing")

    # --- 11. Cleanup ---
    merged = merged.drop(columns=[
        "UnityTime_y", "Time_y", "Time_x", "SubjectID_x", "SubjectID_y", "Subject_x", "Subject_y", "name",
        "marker", "scene_x", "scene_y", "Name", "customevent_col_4", "customevent_col_5",
        "customevent_col_6", "customevent_col_7", "Contents", "Action", "SubjectID"
    ], errors="ignore")

    merged = merged.rename(columns={
        "UnityTime_x": "UnityTime",
        "time": "Frame",
        "Subject": "Participant",
        "slider marker": "anxiety",
        "X_x": "X_pos", "Y_x": "Y_pos", "Z_x": "Z_pos",
        "X_y": "X_rot", "Y_y": "Y_rot", "Z_y": "Z_rot"
    })
    # merged = merged.rename(columns={"SubjectID_x": "SubjectID"})

    for col in merged.columns:
        if any(axis in col for axis in ['_pos', '_rot']):
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    return merged, list(set(warnings))  # ❗warnings 함께 반환
# def extract_agent_df(unity_agent_df, scene_col="scene", time_col="Time", fs=120.0):
#     """
#     Unity Agent 로그에서
#     - scene / Time / AgentName / 위치(x,y,z 등) / 기타 컬럼
#     을 그대로 유지하면서
#     - Scene별로 0부터 시작하는 Frame(integer)을 추가해서 반환.

#     ⚠️ 좌표(x,y,z)는 절대 평균/집계하지 않음.
#     ⚠️ AgentName은 원본 그대로 둠.
#     """
#     import numpy as np
#     import pandas as pd

#     df = unity_agent_df.copy()

#     # 컬럼 이름 정리(공백 제거)
#     df.columns = [c.strip() for c in df.columns]

#     if scene_col not in df.columns:
#         raise ValueError(f"Agent df에 '{scene_col}' 컬럼이 없습니다.")
#     if time_col not in df.columns:
#         raise ValueError(f"Agent df에 '{time_col}' 컬럼이 없습니다.")

#     # 시간 숫자형 변환
#     df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
#     df = df.dropna(subset=[time_col])

#     # Scene별로 Frame 생성
#     frames = []
#     for sc, sub in df.groupby(scene_col):
#         sub = sub.sort_values(time_col).copy()
#         # Scene 시작 시간을 0으로 정규화
#         t0 = sub[time_col].min()
#         # Frame = round( (Time - t0) * fs )
#         frame = ((sub[time_col] - t0) * fs).round().astype(int)
#         frames.append(frame.to_numpy())

#     df["Frame"] = np.concatenate(frames)
#     df = df.drop(columns=['Subject'])

#     # 여기서 groupby / mean 같은 집계 절대 하지 않음
#     # (scene, Frame, AgentName) 기준 중복은 "같은 프레임에 여러 이벤트"로 인정하고 그대로 둔다.

#     return df






# import pandas as pd
# import numpy as np
# import os
# from collections import defaultdict
# ### 🧼 정리된 클렌징 함수: 공통 정제 및 불필요 컬럼 제거

# def extract_customevent_df(unity_ce_df, scene_col="scene", time_col="Time", fs=120.0):
#     """
#     Unity CustomEvent 로그에서
#     - scene / Time / Name / Contents / 기타 컬럼
#     을 유지하면서
#     - Scene별로 0부터 시작하는 Frame(integer)을 추가해서 반환.

#     ⚠️ 여기서는 Subject 컬럼을 건드리지 않음 (삭제는 저장 단계에서 처리).
#     """
#     import numpy as np
#     import pandas as pd

#     df = unity_ce_df.copy()
#     df.columns = [c.strip() for c in df.columns]

#     if scene_col not in df.columns:
#         raise ValueError(f"Customevent df에 '{scene_col}' 컬럼이 없습니다.")
#     if time_col not in df.columns:
#         raise ValueError(f"Customevent df에 '{time_col}' 컬럼이 없습니다.")

#     df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
#     df = df.dropna(subset=[time_col])

#     frames = []
#     for sc, sub in df.groupby(scene_col):
#         sub = sub.sort_values(time_col).copy()
#         t0 = sub[time_col].min()
#         frame = ((sub[time_col] - t0) * fs).round().astype(int)
#         frames.append(frame.to_numpy())

#     df["Frame"] = np.concatenate(frames)
#     df = df.drop(columns=['Subject'], errors='ignore')
#     return df


def extract_agent_df(unity_dict, participant_name, scenes, target_hz=120, main_df=None):
    """
    Unity position/rotation/subjects에서 Agent 트랙을 추출하여

    - AgentName: Unity subjects.Name 그대로
    - SubjectID: Unity의 고유 ID 그대로 유지
    - X_pos, Y_pos, Z_pos: position 좌표
    - X_rot, Y_rot, Z_rot: rotation 좌표
    - scene 내부 Time 0초 기준 → Frame(120Hz) 생성
    - 각 (scene, SubjectID)별로 Frame 0..max까지 연속이 되도록 reindex + ffill

    을 수행한 DataFrame을 반환합니다.
    """

    import pandas as pd
    import numpy as np

    all_agents = []

    for scene in scenes:
        has_pos = scene in unity_dict.get("position", {})
        has_sub = scene in unity_dict.get("subjects", {})
        has_rot = scene in unity_dict.get("rotation", {})

        if not (has_pos and has_sub):
            # subjects 또는 position 자체가 없으면 스킵
            continue

        pos_df = unity_dict["position"][scene].copy()
        subjects_df = unity_dict["subjects"][scene].copy()
        rot_df = None
        if has_rot:
            rot_df = unity_dict["rotation"][scene].copy()

        # ------------------------------------------------
        # 1) 컬럼/값 공백 정리
        # ------------------------------------------------
        pos_df.columns = [c.strip() for c in pos_df.columns]
        subjects_df.columns = subjects_df.columns.str.strip()
        subjects_df["Name"] = subjects_df["Name"].astype(str).str.strip()

        if rot_df is not None:
            rot_df.columns = [c.strip() for c in rot_df.columns]

        # ------------------------------------------------
        # 2) Player / Head 제거 (SubjectID 기준)
        # ------------------------------------------------
        try:
            player_id = subjects_df.loc[subjects_df["Name"] == "Player_VR", "ID"].values[0]
            head_id   = subjects_df.loc[subjects_df["Name"] == "HeadCollider", "ID"].values[0]
        except IndexError:
            print(f"⚠️ Failed to extract head_id or player_id for {participant_name} scene {scene}")
            continue

        pos_df = pos_df[
            (pos_df["SubjectID"] != player_id) &
            (pos_df["SubjectID"] != head_id)
        ].copy()

        # ------------------------------------------------
        # 3) position 컬럼명 → X_pos, Y_pos, Z_pos 강제
        # ------------------------------------------------
        pos_rename = {}
        if "X" in pos_df.columns: pos_rename["X"] = "X_pos"
        if "Y" in pos_df.columns: pos_rename["Y"] = "Y_pos"
        if "Z" in pos_df.columns: pos_rename["Z"] = "Z_pos"
        # 이미 X_pos 형태라면 그대로 사용
        pos_df = pos_df.rename(columns=pos_rename)

        # ------------------------------------------------
        # 4) rotation 있으면 position과 merge
        #     - Time + SubjectID 기준
        # ------------------------------------------------
        if rot_df is not None:
            # rotation 쪽 SubjectID/Time 정리
            if "SubjectID" not in rot_df.columns and "ID" in rot_df.columns:
                rot_df = rot_df.rename(columns={"ID": "SubjectID"})

            # rotation 컬럼명을 X_rot, Y_rot, Z_rot로 강제
            rot_rename = {}
            if "X" in rot_df.columns: rot_rename["X"] = "X_rot"
            if "Y" in rot_df.columns: rot_rename["Y"] = "Y_rot"
            if "Z" in rot_df.columns: rot_rename["Z"] = "Z_rot"
            rot_df = rot_df.rename(columns=rot_rename)

            # 필요한 컬럼만 남기고 merge
            rot_keep_cols = ["Time", "SubjectID"]
            for c in ["X_rot", "Y_rot", "Z_rot"]:
                if c in rot_df.columns:
                    rot_keep_cols.append(c)
            rot_df = rot_df[rot_keep_cols].copy()

            # Time numeric 변환
            pos_df["Time"] = pd.to_numeric(pos_df["Time"], errors="coerce")
            rot_df["Time"] = pd.to_numeric(rot_df["Time"], errors="coerce")
            pos_df = pos_df.dropna(subset=["Time"])
            rot_df = rot_df.dropna(subset=["Time"])

            pos_df = pos_df.merge(
                rot_df,
                how="left",
                on=["Time", "SubjectID"]
            )
        else:
            # rotation이 전혀 없을 때를 대비해 Time만 numeric 처리
            pos_df["Time"] = pd.to_numeric(pos_df["Time"], errors="coerce")
            pos_df = pos_df.dropna(subset=["Time"])

        # ------------------------------------------------
        # 5) scene 컬럼 추가 + AgentName 붙이기
        # ------------------------------------------------
        pos_df["scene"] = scene

        df = pos_df.merge(
            subjects_df[["ID", "Name"]],
            how="left",
            left_on="SubjectID",
            right_on="ID"
        )
        df.drop(columns=["ID"], inplace=True)
        df.rename(columns={"Name": "AgentName"}, inplace=True)

        all_agents.append(df)

    # ------------------------------------------------
    # 6) 모든 scene concat
    # ------------------------------------------------
    if not all_agents:
        print(f"🚨 No agent data found for participant {participant_name} in scenes {scenes}.")
        return pd.DataFrame()

    agent_df = pd.concat(all_agents, ignore_index=True)

    # ------------------------------------------------
    # 7) Time → Frame (scene별 0부터, 120Hz 기준)
    # ------------------------------------------------
    agent_df["Frame"] = -1
    for sc, idx in agent_df.groupby("scene").groups.items():
        sub = agent_df.loc[idx].sort_values("Time")
        t0 = sub["Time"].min()
        f = ((sub["Time"] - t0) * target_hz).round().astype(int)
        agent_df.loc[sub.index, "Frame"] = f.values

    # Time은 이후 안 쓰면 제거
    agent_df.drop(columns=["Time"], inplace=True)

    # ------------------------------------------------
    # 8) (scene, SubjectID)별로 Frame 연속화 & ffill
    # ------------------------------------------------
    filled = []
    for (sc, sid), sub in agent_df.groupby(["scene", "SubjectID"]):
        sub = sub.sort_values("Frame")

        # 같은 Frame 중복 → 첫 번째만 유지
        dup_mask = sub["Frame"].duplicated()
        if dup_mask.any():
            n_dup = dup_mask.sum()
            print(
                f"⚠️ [Agent] duplicate Frame after quantization: "
                f"scene={sc}, SubjectID={sid}, n={n_dup} → first row kept"
            )
            sub = sub[~dup_mask]

        sub = sub.set_index("Frame")
        full_range = range(sub.index.min(), sub.index.max() + 1)
        sub = sub.reindex(full_range)

        sub["scene"] = sc
        sub["SubjectID"] = sid
        if "AgentName" in sub.columns:
            sub["AgentName"] = sub["AgentName"].ffill()

        # 좌표/회전값 포함해서 전부 ffill로 채우기
        sub = sub.ffill()

        sub = sub.reset_index().rename(columns={"index": "Frame"})
        filled.append(sub)

    agent_df = pd.concat(filled, ignore_index=True)

    # 최종 QC: (scene, Frame, SubjectID) 기준 중복 체크
    dup_cnt = agent_df.duplicated(subset=["scene", "Frame", "SubjectID"]).sum()
    if dup_cnt > 0:
        print(f"⚠️ Agent duplicated rows detected (scene, Frame, SubjectID): {dup_cnt}")

    return agent_df






import pandas as pd
import numpy as np
import os
from collections import defaultdict
### 🧼 정리된 클렌징 함수: 공통 정제 및 불필요 컬럼 제거

def extract_customevent_df(
    unity_dict,
    scenes,
    target_hz=120,
    participant_name="unknown",
    main_df=None,
):
    """
    Unity customevent 로그에서 scene / Time / Name / Contents 등을 유지하면서,
    각 scene 내부 시간을 0초 기준으로 정규화한 뒤
    120 Hz 기준 Frame(정수)을 만들어 반환.

    - Frame: scene별 0부터 시작 (Main, Agent와 동일 컨벤션)
    - 같은 Frame에 여러 이벤트(행) 존재 가능 → 그대로 유지
    - SubjectID / UnityTime / Time은 드롭
    - 문자열 전체를 깨는 aggressive string-cleaning은 제거
    """

    import pandas as pd
    import numpy as np

    # 1. Concatenate all relevant scenes
    dfs = []
    for s in scenes:
        if "customevent" in unity_dict and s in unity_dict["customevent"]:
            dfs.append(unity_dict["customevent"][s].copy())
    if not dfs:
        print(f"🚨 No customevent data found for participant {participant_name} in scenes {scenes}.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # 2. 기존 클렌징 로직이 너무 aggressive 했다면 최소화해서 사용
    df = clean_unity(df, dtype="customevent")
    df = strip_and_filter(df, scenes=scenes)

    # 3. Time → 숫자형
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    # 4. scene별로 0초 기준 정규화 + Frame(120 Hz) 생성
    frames_all = []
    for sc, sub in df.groupby("scene"):
        sub = sub.sort_values("Time").copy()
        t0 = sub["Time"].min()
        # scene 내부 시간(초)
        t_rel = sub["Time"] - t0
        # 120 Hz 기준 Frame
        f = (t_rel * target_hz).round().astype(int)
        frames_all.append(f.to_numpy())
    df["Frame"] = np.concatenate(frames_all)

    # (옵션) main_df가 있으면 duration sanity-check만 수행
    if main_df is not None and "scene" in main_df.columns and "Frame" in main_df.columns:
        for sc in scenes:
            ce_sub = df[df["scene"] == sc]
            main_sub = main_df[main_df["scene"] == sc]
            if ce_sub.empty or main_sub.empty:
                continue
            dur_ce = ce_sub["Frame"].max() - ce_sub["Frame"].min()
            dur_main = main_sub["Frame"].max() - main_sub["Frame"].min()
            if abs(dur_ce - dur_main) > int(0.5 * target_hz):  # 0.5초 이상 차이 나면 경고
                print(
                    f"⚠️ [customevent] duration mismatch in {participant_name}, scene={sc}: "
                    f"CE≈{dur_ce/target_hz:.2f}s vs Main≈{dur_main/target_hz:.2f}s"
                )

    # 5. 의미 있는 컬럼들 rename (이벤트 타입/이름 구분)
    df = df.rename(columns={"Name": "EventType", "Contents": "Name"})

    # 6. Column cleanup: 시간 관련 원본 컬럼만 제거
    for col in ["UnityTime", "Time", "SubjectID"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


    return df

def load_filtered_rsp_dataframe(filtered_rsp_dir: str, participant_name: str):
    """
    필터된 VR.acq 파일에서 **RSP 신호만** 꺼내오는 헬퍼.

    - 입력
      filtered_rsp_dir : 필터된 RSP .acq들이 있는 폴더 경로
      participant_name : 원본 참가자 폴더명 (예: "bhk9709")

    - 출력
      rsp_df : 최소한의 컬럼만 가진 DataFrame
        - 가능한 경우: ['Sample', 'Time (s)', 'RSP']
        - 최소 보장: 인덱스 + 'RSP' 하나
      → scene / marker 정보는 포함하지 않음 (이후 원본 VR + Unity 기준으로 다시 라벨링할 것)

      파일이 없거나 RSP 채널을 찾지 못하면 None 반환.
    """
    import os
    import traceback
    import pandas as pd
    # 같은 파일 안에 정의된 함수이므로 그냥 이름으로 호출해도 됨
    # (외부 모듈이라면 from Dataloader import read_acqknowledge_with_markers 필요)
    from Dataloader import read_acqknowledge_with_markers

    try:
        file_path = os.path.join(filtered_rsp_dir, f"{participant_name}_VR.acq")
        if not os.path.exists(file_path):
            print(f"⚠️ [load_filtered_rsp_dataframe] No filtered RSP file for {participant_name}: {file_path}")
            return None

        # Biopac 파서로 전체 신호 로딩 (marker/scene은 여기서는 무시)
        df, _, _ = read_acqknowledge_with_markers(file_path)
        df.columns = df.columns.str.strip()

        # (안전장치) 예전에 쓰던 채널 이름 매핑 한 번 더 시도
        df.rename(columns={
            "RSP, X, RSPEC-R": "RSP1",
            "RSP, X, RSPEC": "RSP2",
            "EDA, Y, PPGED-R": "EDA",
            "PPG, X, PPGED-R": "PPG",
        }, inplace=True)

        # 🔍 RSP 후보 채널 탐색
        rsp_cols = [col for col in df.columns if col.upper().startswith("RSP")]
        if not rsp_cols:
            print(f"⚠️ [load_filtered_rsp_dataframe] No RSP-like columns for {participant_name}. cols={list(df.columns)}")
            return None

        # 보통 필터된 채널이 RSP2라서 먼저 우선
        if "RSP2" in rsp_cols:
            rsp_col = "RSP2"
        else:
            # 없다면 마지막 RSP 계열을 필터된 것으로 가정
            rsp_col = rsp_cols[-1]

        # ✅ 최소한의 정보만 담은 DataFrame 구성
        out_cols = []
        for c in ["Sample", "Time (s)", "Time"]:
            if c in df.columns:
                out_cols.append(c)

        if out_cols:
            rsp_df = df[out_cols].copy()
        else:
            # 샘플/시간 정보가 없으면 인덱스만 사용
            rsp_df = pd.DataFrame(index=df.index)

        rsp_df["RSP"] = df[rsp_col].astype("float32")

        return rsp_df

    except Exception as e:
        print(f"❌ [load_filtered_rsp_dataframe] Error for {participant_name}: {e}")
        traceback.print_exc()
        return None
    
def sync_biopac_markers_with_unity(
    df_raw,
    event_markers_df,
    unity_dict,
    scenes,
    fs: int = 2000,
    unity_hz: int = 60,
):
    """
    Unity 이벤트를 이용해 Biopac StartP/End marker 누락을 보정하고,
    df_raw에 scene / marker("Ongoing") 라벨을 다시 붙인다.
    """
    import numpy as np
    import pandas as pd

    def _norm(s):
        """공백/탭/CR 제거 + 콤마/언더스코어/하이픈 제거 + 소문자화."""
        if not isinstance(s, str):
            s = str(s)
        return (
            s.lower()
            .replace("\r", "")
            .replace("\t", "")
            .replace("\xa0", "")
            .strip()
            .replace(",", "")
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )


    df = df_raw.copy()

    # 0) 인덱스/시간 정리
    if "Sample" in df.columns:
        try:
            df.index = df["Sample"].astype(int)
        except Exception:
            pass

    em = event_markers_df.copy()
    if "Sample" in em.columns:
        try:
            em.index = em["Sample"].astype(int)
        except Exception:
            pass

    if "Time (s)" not in em.columns:
        em["Time (s)"] = em.index.to_series() / float(fs)

    if "scene" not in em.columns:
        em["scene"] = np.nan
    if "marker" not in em.columns:
        em["marker"] = np.nan

    # 1) Unity 쪽 데이터 준비
    # 1-1) customevent key 찾기
    ce_key = None
    for k in unity_dict.keys():
        try:
            k_low = k.lower()
        except Exception:
            continue
        if k_low.startswith("customevent"):
            ce_key = k
            break

    ce_list = []
    if ce_key is not None:
        for sc in scenes:
            if sc in unity_dict.get(ce_key, {}):
                df_ce = unity_dict[ce_key][sc].copy()
                # 🔧 컬럼 클린업 (빈 이름 → customevent_col_i 등)
                df_ce = clean_unity(df_ce, dtype="customevent")
                df_ce["scene"] = sc
                ce_list.append(df_ce)

    if not ce_list:
        print("⚠️ [sync] No customevent data found → skip marker sync.")
        df["scene"] = em["scene"].reindex(df.index)
        df["marker"] = em["marker"].reindex(df.index)
        return df, pd.DataFrame()

    ce_all = pd.concat(ce_list, ignore_index=True)
    ce_all.columns = [c.strip() for c in ce_all.columns]

    if "Time" not in ce_all.columns:
        print("⚠️ [sync] No 'Time' column in customevent → skip marker sync.")
        df["scene"] = em["scene"].reindex(df.index)
        df["marker"] = em["marker"].reindex(df.index)
        return df, pd.DataFrame()

    ce_all["Time"] = pd.to_numeric(ce_all["Time"], errors="coerce")
    ce_all = ce_all.dropna(subset=["Time"])

    # 1-2) lifecycleevents key 찾기
    le_key = None
    for k in unity_dict.keys():
        try:
            k_low = k.lower()
        except Exception:
            continue
        if "lifecycle" in k_low:
            le_key = k
            break

    # 1-3) Scene별 Unity Start/End 계산
    unity_markers = {}

    # 🔍 전체 unity_dict 키도 한 번 찍어두기 (원하면 주석 처리)
    print(f"[sync-debug] unity_dict keys = {list(unity_dict.keys())}")
    print(f"[sync-debug] ce_key={ce_key}, le_key={le_key}")

    for sc in scenes:
        try:
            # --- QC용 보조 변수들 초기화 ---
            t_start_custom = None          # ANY StartP (customevent 기반)
            t_start_lifecycle = None       # LifeCycleEvents Start
            t_end_goalzone_first = None    # 첫 GoalZone 시간
            t_end_goalzone_last = None     # 마지막 GoalZone 시간
            t_end_goalzone = None          # 세그먼트에 실제로 쓸 GoalZone 시간 (last)
            t_end_custom = None            # customevent 'end' 시간 (SceneStatus 기반)

            ce_sub = ce_all[ce_all["scene"] == sc].copy()
            ce_sub.columns = ce_sub.columns.str.strip()

            print(f"[sync-debug][{sc}] ce_sub shape={ce_sub.shape}, columns={list(ce_sub.columns)}")

            # --------------------------------------
            # 공통: Subject / Name / Contents 정규화
            # --------------------------------------
            subj_norm = None
            if "Subject" in ce_sub.columns:
                subj_norm = ce_sub["Subject"].astype(str).map(_norm)

            name_norm = None
            if "Name" in ce_sub.columns:
                name_norm = ce_sub["Name"].astype(str).map(_norm)

            contents_low = None
            if "Contents" in ce_sub.columns:
                contents_low = ce_sub["Contents"].astype(str).str.lower()

            # ✅ payload: Contents + customevent_col_4~7 모두 합치기
            payload_cols = []
            for col in ["Contents", "customevent_col_4", "customevent_col_5",
                        "customevent_col_6", "customevent_col_7"]:
                if col in ce_sub.columns:
                    payload_cols.append(col)

            payload = None
            if payload_cols:
                payload = (
                    ce_sub[payload_cols]
                    .astype(str)
                    .agg(" ".join, axis=1)
                    .str.lower()
                )

            # -------------------
            # ① StartP 탐색 (1순위: customevent, 2순위: LifeCycle)
            # -------------------
            t_start = None
            start_source = "none"

            # ①-1) customevent SceneStatus + payload에 'startp' 포함
            if payload is not None and "Name" in ce_sub.columns:
                if name_norm is None:
                    name_norm = ce_sub["Name"].astype(str).map(_norm)

                m_sp = ce_sub[name_norm.str.contains("scenestatus", na=False)].copy()
                mask_sp = payload.loc[m_sp.index].str.contains("startp", na=False)
                m_sp = m_sp[mask_sp]

                print(f"[sync-debug][{sc}] customevent StartP candidates (SceneStatus+payload) = {len(m_sp)}")

                if not m_sp.empty:
                    t_start = float(m_sp["Time"].min())
                    start_source = "CustomEvent(SceneStatus & payload contains 'StartP')"

            # ①-2) fallback: payload 어디든 'startp' 들어간 첫 이벤트
            if t_start is None and payload is not None:
                m_sp2 = ce_sub[payload.str.contains("startp", na=False)].copy()
                print(f"[sync-debug][{sc}] customevent StartP candidates (payload only) = {len(m_sp2)}")
                if not m_sp2.empty:
                    t_start = float(m_sp2["Time"].min())
                    start_source = "CustomEvent(payload contains 'StartP')"

            # -------------------
            # ①-보너스) QC용: ANY StartP (payload 우선, 그다음 Name/Contents)
            #   - segmentation에는 사용하지 않고, dur_unity_custom_startp_end 계산에만 사용
            # -------------------
            if t_start_custom is None:
                # (a) payload 우선
                if payload is not None:
                    m_any_idx = payload.str.contains("startp", case=False, na=False)
                    m_any = ce_sub[m_any_idx]
                    print(f"[sync-debug][{sc}] customevent ANY StartP candidates (payload) = {len(m_any)}")
                    if not m_any.empty:
                        t_start_custom = float(m_any["Time"].min())

                # (b) Name/Contents fallback
                if (t_start_custom is None) and (("Name" in ce_sub.columns) or (contents_low is not None)):
                    m_any = ce_sub.copy()
                    mask_any = pd.Series(False, index=ce_sub.index)
                    if name_norm is not None:
                        mask_any = mask_any | name_norm.str.contains("startp", na=False)
                    if contents_low is not None:
                        mask_any = mask_any | contents_low.str.contains("startp", na=False)
                    m_any = ce_sub[mask_any]
                    print(f"[sync-debug][{sc}] customevent ANY StartP candidates (Name/Contents) = {len(m_any)}")
                    if not m_any.empty:
                        t_start_custom = float(m_any["Time"].min())

            # -------------------
            # ② End 탐색 (1순위: customevent 'end', 2순위: GoalZone 마지막)
            # -------------------
            t_end = None
            end_source = "none"
            # QC용: GoalZone 기반 End, customevent 기반 end도 따로 보관
            t_end_custom = None
            t_end_goalzone_first = None
            t_end_goalzone_last = None
            t_end_goalzone = None

            # ②-1) customevent payload 기반 'end' (SceneStatus, 'startp'는 제외)
            if payload is not None:
                m_end = ce_sub.copy()

                # 가능하면 SceneStatus 행만
                if "Name" in ce_sub.columns:
                    if name_norm is None:
                        name_norm = ce_sub["Name"].astype(str).map(_norm)
                    m_end = m_end[name_norm.loc[m_end.index].str.contains("scenestatus", na=False)]

                if not m_end.empty:
                    mask_end = payload.loc[m_end.index].str.contains("end", na=False) & \
                               ~payload.loc[m_end.index].str.contains("startp", na=False)
                    m_end = m_end[mask_end]

                print(f"[sync-debug][{sc}] customevent 'end' candidates (SceneStatus+payload) = {len(m_end)}")

                if not m_end.empty:
                    t_end = float(m_end["Time"].min())
                    t_end_custom = t_end
                    end_source = "CustomEvent(SceneStatus 'end')"

            # ②-2) GoalZone 기반 fallback (End가 아직 없을 때만)
            if (t_end is None) and ("Name" in ce_sub.columns):
                name_norm = ce_sub["Name"].astype(str).map(_norm)
                goal_mask = name_norm == _norm("GoalZone")
                m_goal = ce_sub[goal_mask]
                print(f"[sync-debug][{sc}] GoalZone(Name) candidates = {len(m_goal)}")
                if not m_goal.empty:
                    # QC용: 첫/마지막 GoalZone 둘 다 저장
                    t_end_goalzone_first = float(m_goal["Time"].min())
                    t_end_goalzone_last  = float(m_goal["Time"].max())

                    # 👉 세그먼트 끝으로는 "마지막 GoalZone" 사용
                    t_end = t_end_goalzone_last
                    t_end_goalzone = t_end_goalzone_last
                    end_source = "CustomEvent(Name=='GoalZone' last)"

            # ②-2b) Name이 없고 Contents만 있을 경우 GoalZone fallback
            if (t_end is None) and (contents_low is not None) and ("Name" not in ce_sub.columns):
                m_goal2 = ce_sub[contents_low.str.contains("goalzone", na=False)]
                print(f"[sync-debug][{sc}] GoalZone(Contents) candidates = {len(m_goal2)}")
                if not m_goal2.empty:
                    t_end_goalzone_first = float(m_goal2["Time"].min())
                    t_end_goalzone_last  = float(m_goal2["Time"].max())
                    t_end = t_end_goalzone_last
                    t_end_goalzone = t_end_goalzone_last
                    end_source = "CustomEvent(Contents contains 'GoalZone' last)"

            # -------------------
            # ③ LifeCycleEvents fallback (Start) – customevent StartP를 못 찾았을 때만
            # -------------------
            if t_start is None and le_key is not None and sc in unity_dict.get(le_key, {}):
                print(f"[sync-debug][{sc}] try LifeCycle fallback: le_key={le_key}")

                le_sub = unity_dict[le_key][sc].copy()
                le_sub.columns = le_sub.columns.str.strip()

                # ③-1) Player_VR ID 찾기
                player_id = None

                # (a) subjects 스트림에서 찾기
                subjects_key = "subjects"
                if subjects_key in unity_dict and sc in unity_dict[subjects_key]:
                    subj_df = unity_dict[subjects_key][sc].copy()
                    subj_df.columns = subj_df.columns.str.strip()
                    name_cols = [c for c in subj_df.columns if c.lower() in ("name", "subjectname")]
                    id_cols = [c for c in subj_df.columns if c.lower() in ("subjectid", "id")]
                    if name_cols and id_cols:
                        name_col = name_cols[0]
                        id_col = id_cols[0]
                        subj_df[name_col] = subj_df[name_col].astype(str).map(_norm)
                        target_name = _norm("Player_VR")
                        m_player = subj_df[subj_df[name_col] == target_name]
                        print(f"[sync-debug][{sc}] subjects stream Player_VR rows = {len(m_player)}")
                        if not m_player.empty:
                            player_id = m_player[id_col].iloc[0]

                # (b) customevent.Subject에서 찾기 (혹시 subjects 스트림에 없을 경우)
                if player_id is None and subj_norm is not None:
                    idxs = ce_sub[subj_norm == _norm("Player_VR")]["SubjectID"].dropna()
                    print(f"[sync-debug][{sc}] customevent Player_VR SubjectID candidates = {list(idxs.unique())}")
                    if not idxs.empty:
                        player_id = idxs.iloc[0]

                # ③-2) LifeCycleEvents에서 Start 찾기
                if "Time" in le_sub.columns and "Event" in le_sub.columns:
                    le_sub["Time"] = pd.to_numeric(le_sub["Time"], errors="coerce")
                    le_sub = le_sub.dropna(subset=["Time"])

                    if player_id is not None:
                        subj_cols = [c for c in le_sub.columns if c.lower() in ("subjectid", "id")]
                        if subj_cols:
                            subj_col = subj_cols[0]
                            le_sub = le_sub[le_sub[subj_col] == player_id]
                            print(f"[sync-debug][{sc}] LifeCycle filtered by Player_VR ID={player_id}, rows={len(le_sub)}")
                    else:
                        print(f"[sync-debug][{sc}] LifeCycle fallback with NO player_id (using all Start events)")

                    event_norm = le_sub["Event"].astype(str).map(_norm)
                    m_start = le_sub[event_norm.str.contains("start", na=False)]

                    print(f"[sync-debug][{sc}] LifeCycle Start rows = {len(m_start)}")

                    if not m_start.empty:
                        t_start = float(m_start["Time"].min())
                        t_start_lifecycle = t_start   # ✅ QC용 저장
                        if player_id is not None:
                            start_source = "LifeCycle(Player_VR Start)"
                        else:
                            start_source = "LifeCycle(First Start; no Player_VR ID)"

            # 최종 로그
            print(
                f"[sync-debug][{sc}] Unity StartP_time={t_start} ({start_source}), "
                f"End_time={t_end} ({end_source})"
            )

            unity_markers[sc] = {
                "StartP_time": t_start,
                "End_time": t_end,
                "StartP_source": start_source,
                "End_source": end_source,
                "StartP_time_custom": t_start_custom,
                "End_time_custom": t_end_custom,
                "StartP_time_lifecycle": t_start_lifecycle,
                "End_time_goalzone": t_end_goalzone,
            }

        except Exception as e:
            print(f"⚠️ [sync-error] while processing Unity markers for scene={sc}: {repr(e)}")
            raise


    # 2) Biopac markers 정리
    biopac_markers = {}
    for sc in scenes:
        sub = em[em["scene"] == sc]
        try:
            if sub.empty:
                biopac_markers[sc] = {
                    "StartP_sample": None,
                    "End_sample": None,
                    "StartP_time": None,
                    "End_time": None,
                }
                continue

            marker_series = sub["marker"].astype(str)
            if hasattr(marker_series, "str"):
                marker_low = marker_series.str.lower()
            else:
                print(f"[sync-debug][{sc}] marker column is not string-like Series → type={type(marker_series)}")
                marker_low = marker_series  # 어차피 비교에서 안맞을 것

            start_rows = sub[marker_low == "startp"]
            end_rows = sub[marker_low == "end"]

            if not start_rows.empty:
                start_sample = int(start_rows.index[0])
                start_time = float(start_rows["Time (s)"].iloc[0])
            else:
                start_sample = None
                start_time = None

            if not end_rows.empty:
                end_sample = int(end_rows.index[0])
                end_time = float(end_rows["Time (s)"].iloc[0])
            else:
                end_sample = None
                end_time = None

            biopac_markers[sc] = {
                "StartP_sample": start_sample,
                "End_sample": end_sample,
                "StartP_time": start_time,
                "End_time": end_time,
            }
        except Exception as e:
            print(f"⚠️ [sync-error] while reading Biopac markers for scene={sc}: {repr(e)}")
            raise

    # 3) Scene별 보정 (A/B/C 케이스)
    scene_intervals = {}
    qc_records = []

    # ✅ missing_type / imputed_flag를 인자로 받는 버전으로 통일
    def _record(scene, method, start, end, tsb, teb, tsu, teu, missing_type, imputed_flag, note=""):
        if (start is not None) and (end is not None):
            dur_b = (end - start) / float(fs)
        else:
            dur_b = None

        if (tsu is not None) and (teu is not None):
            dur_u = teu - tsu
        else:
            dur_u = None

        if (dur_b is not None) and (dur_u is not None):
            diff = abs(dur_b - dur_u)
        else:
            diff = None

        qc_records.append({
            "scene": scene,
            "method": method,
            "missing_type": missing_type,   # StartP / End / Both / None
            "imputed_flag": imputed_flag,   # 0=original, 1=unity-based imputation
            "start_sample": start,
            "end_sample": end,
            "biopac_start_time": tsb,
            "biopac_end_time": teb,
            "unity_start_time": tsu,
            "unity_end_time": teu,
            "dur_biopac": dur_b,
            "dur_unity": dur_u,
            "dur_diff": diff,
            "note": note
        })

    imputed_count = {"StartP": 0, "End": 0, "Both": 0}

    for sc in scenes:
        b = biopac_markers.get(sc, {})
        u = unity_markers.get(sc, {})

        bs = b.get("StartP_sample")
        be = b.get("End_sample")
        ts_b = b.get("StartP_time")
        te_b = b.get("End_time")

        ts_u = u.get("StartP_time")
        te_u = u.get("End_time")

        final_start = bs
        final_end = be
        method = "original"

        # ✅ 기본값: 결측 없음 / Imputation 안함
        missing_type = "None"
        imputed_flag = 0

        # A) StartP 없음 + End 있음 + Unity duration 있음
        if (bs is None) and (be is not None) and (ts_u is not None) and (te_u is not None):
            missing_type = "StartP"
            imputed_flag = 1
            dur_u = te_u - ts_u
            if te_b is None:
                te_b = be / float(fs)
            ts_est = te_b - dur_u
            final_start = int(round(ts_est * fs))
            method = "unity_duration_from_end"
            ts_b = ts_est
            imputed_count["StartP"] += 1

        # B) StartP 있음 + End 없음 + Unity duration 있음
        elif (bs is not None) and (be is None) and (ts_u is not None) and (te_u is not None):
            missing_type = "End"
            imputed_flag = 1
            dur_u = te_u - ts_u
            if ts_b is None:
                ts_b = bs / float(fs)
            te_est = ts_b + dur_u
            final_end = int(round(te_est * fs))
            method = "unity_duration_from_start"
            te_b = te_est
            imputed_count["End"] += 1

        # C) 둘 다 없음 → prev/next scene + Unity 비율로 복원 시도
        elif (bs is None) and (be is None):
            missing_type = "Both"
            imputed_flag = 1
            method = "unity_neighbors"
            imputed_count["Both"] += 1

            try:
                idx = scenes.index(sc)
            except ValueError:
                idx = -1

            prev_scene = scenes[idx - 1] if idx > 0 else None
            next_scene = scenes[idx + 1] if (idx >= 0 and idx + 1 < len(scenes)) else None

            success = False

            if prev_scene and next_scene:
                b_prev = biopac_markers.get(prev_scene, {})
                b_next = biopac_markers.get(next_scene, {})
                u_prev = unity_markers.get(prev_scene, {})
                u_next = unity_markers.get(next_scene, {})

                prev_end_sample = b_prev.get("End_sample")
                next_start_sample = b_next.get("StartP_sample")

                prev_end_time_u   = u_prev.get("End_time")
                this_start_time_u = u.get("StartP_time")
                this_end_time_u   = u.get("End_time")
                next_start_time_u = u_next.get("StartP_time")

                if (prev_end_sample is not None) and (next_start_sample is not None) and \
                   (prev_end_time_u is not None) and (this_start_time_u is not None) and \
                   (this_end_time_u is not None) and (next_start_time_u is not None):

                    total_b = (next_start_sample - prev_end_sample) / float(fs)
                    d1 = this_start_time_u - prev_end_time_u
                    d2 = this_end_time_u - this_start_time_u
                    d3 = next_start_time_u - this_end_time_u
                    s = d1 + d2 + d3
                    if s > 0:
                        r1 = d1 / s
                        r2 = d2 / s
                        r3 = d3 / s

                        t_prev_end_b = prev_end_sample / float(fs)

                        t_start_est = t_prev_end_b + r1 * total_b
                        t_end_est   = t_prev_end_b + (r1 + r2) * total_b

                        final_start = int(round(t_start_est * fs))
                        final_end   = int(round(t_end_est * fs))
                        ts_b = t_start_est
                        te_b = t_end_est
                        success = True

            if not success:
                method = "unity_neighbors_failed"
                final_start = None
                final_end = None

        # 이상 방어: start>=end
        if (final_start is not None) and (final_end is not None) and (final_start >= final_end):
            _record(
                sc, method, final_start, final_end,
                ts_b, te_b, ts_u, te_u,
                missing_type, imputed_flag,
                note="start>=end → invalid, ignored",
            )
            scene_intervals[sc] = {"start": None, "end": None}
            continue

        scene_intervals[sc] = {"start": final_start, "end": final_end}
        _record(
            sc, method, final_start, final_end,
            ts_b, te_b, ts_u, te_u,
            missing_type, imputed_flag,
        )

    imputed_df = pd.DataFrame(qc_records)

    # ----- 추가 QC: Unity duration 여러 버전 계산 -----
    if len(imputed_df) > 0:
        imputed_df = imputed_df.copy()
        if "dur_unity" in imputed_df.columns:
            imputed_df.rename(
                columns={
                    "dur_unity": "dur_unity_lifecycle_start_goalzone",
                    "dur_diff": "dur_diff_lifecycle",
                },
                inplace=True,
            )

        import numpy as np

        imputed_df["dur_unity_custom_startp_end"] = np.nan
        imputed_df["dur_diff_custom_startp_end"] = np.nan

        for idx, row in imputed_df.iterrows():
            sc = row["scene"]
            u = unity_markers.get(sc, {})
            t_sc = u.get("StartP_time_custom")
            t_ec = u.get("End_time_custom")
            if (t_sc is not None) and (t_ec is not None):
                dur_c = t_ec - t_sc
                imputed_df.at[idx, "dur_unity_custom_startp_end"] = dur_c
                dur_b = row.get("dur_biopac")
                if pd.notna(dur_b):
                    imputed_df.at[idx, "dur_diff_custom_startp_end"] = abs(dur_b - dur_c)

    # 4) Scene / marker 재라벨링
    df["scene"] = np.nan
    df["marker"] = np.nan

    for sc in scenes:
        interval = scene_intervals.get(sc, {})
        s = interval.get("start")
        e = interval.get("end")
        if (s is None) or (e is None) or (s >= e):
            continue

        mask = (df.index >= s) & (df.index <= e)
        df.loc[mask, "scene"] = sc
        df.loc[mask, "marker"] = "Ongoing"

    # 요약 로그
    total_imputed = sum(imputed_count.values())
    print(
        f"[sync] {total_imputed} scene(s) imputed → "
        f"StartP:{imputed_count['StartP']}  "
        f"End:{imputed_count['End']}  "
        f"Both:{imputed_count['Both']}"
    )

    if len(imputed_df) > 0:
        diff_col = "dur_diff_lifecycle" if "dur_diff_lifecycle" in imputed_df.columns else "dur_diff"
        dur_u_col = "dur_unity_lifecycle_start_goalzone" if "dur_unity_lifecycle_start_goalzone" in imputed_df.columns else "dur_unity"

        bad = imputed_df[imputed_df[diff_col].notna() & (imputed_df[diff_col] > 0.5)]
        if not bad.empty:
            print(f"⚠️ [sync] Large Biopac–Unity duration diff in {len(bad)} scene(s):")
            cols_to_show = [
                "scene",
                "dur_biopac",
                dur_u_col,
            ]
            if "dur_unity_custom_startp_end" in bad.columns:
                cols_to_show.append("dur_unity_custom_startp_end")
            cols_to_show.append(diff_col)
            if "dur_diff_custom_startp_end" in bad.columns:
                cols_to_show.append("dur_diff_custom_startp_end")
            print(bad[cols_to_show])

    return df, imputed_df

def check_scene_files(participant_path, unity_dict, scenes):
    """
    Scene별로 Unity/Physio/Psychopy/Tracker 존재 여부를 bool로 리턴.
    - Unity: customevent OR position 중 하나라도 있으면 있다고 간주 (필요시 더 타이트하게 조정)
    - Physio: VR.acq 파일 유무
    - Psychopy: *anxiety.log 유무
    - Tracker: .xdf 유무
    """
    files = os.listdir(participant_path)
    has_vr_acq = any((f.endswith('.acq') and 'VR' in f and 'BT' not in f) for f in files)
    has_xdf = any(f.endswith('.xdf') for f in files)
    has_anxiety = any(f.endswith('anxiety.log') for f in files)

    rows = []
    for sc in scenes:
        has_unity = (
            (sc in unity_dict.get('customevent', {})) or
            (sc in unity_dict.get('position', {})) or
            (sc in unity_dict.get('rotation', {}))
        )
        rows.append({
            "Scene": sc,
            "has_unity": bool(has_unity),
            "has_biopac": bool(has_vr_acq),
            "has_psychopy": bool(has_anxiety),
            "has_tracker": bool(has_xdf),
        })
    return rows
