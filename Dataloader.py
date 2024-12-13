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
            
    testdata = nk.read_xdf(directory, desc = descs)
    
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

def dataloader(datapath_top: str, scenes: list):
    '''
    This function load data from experiment file folder.
    Data list
    #Anxiety data(.log)-anxiety_psy
    #Transform data(.csv)-position, rotation, customevent in/out
    '''
    #Data loader: Position and rotation data, anxiety data.
    #folder name
    names = []
    #Anxiety data psychopy room
    data = {}
    data["Psychopy"] = []
    #Physiology data room
    data["Physiology"] = []
    data["Tracker"] = []
    data["Eyetracker"] = []
    data["Facetracker"] = []
    
    
    #Transform data
    unity = defaultdict(dict)
    #start time
    zeros = defaultdict(dict)
    pIDs = defaultdict(dict)
    
    ## Create List for the Space tp Store
    datapath = os.path.join(datapath_top, os.listdir(datapath_top)[0])
    for in_folder in os.listdir(datapath):
        formet = in_folder.split(".")[-1]
        # if formet == "xdf":
        #     dtype = in_folder.split("_")[-1].split(".")[0]
        #     data[dtype] = []
        if formet == "csv":
            scene = in_folder.split("_")[0]
            dtype = in_folder.split("_")[1].split(".")[0]
            unity[dtype][scene] = []
            pIDs[dtype][scene] = []
            
   
    for folder in os.listdir(datapath_top):        
        datapath = os.path.join(datapath_top, folder)
        names.append(folder)
        print(folder)
        #anxiety data
        for log in glob(datapath+ '/*.log'):
            data["Psychopy"].append(log)
            
        #Transform data
        _allpos = pd.DataFrame()
        _allrot = pd.DataFrame()
        _allcus = pd.DataFrame()
        
        for in_folder in os.listdir(datapath):
            formet = in_folder.split(".")[-1]
            if formet == "acq":
                dtype = in_folder.split("_")[-1].split(".")[0]
                if dtype == "VR":
                    _,_,_,result = read_acqknowledge_with_markers(os.path.join(datapath, in_folder))
                    data["Physiology"].append(result)
            if formet == "xdf":
                dtype = "Tracker"
                data[dtype].append(xdfreaderfixer(os.path.join(datapath, in_folder)))
            if formet == "csv":
                scene = in_folder.split("_")[0]
                dtype = in_folder.split("_")[1].split(".")[0]
                
                if dtype == "customevent":
                    _df = pd.read_csv(datapath+'/'+in_folder,
                            engine='python',
                            encoding='utf-8',
                            names = ['ID','Time','Action', 'Actor', '1','2','3','4']
                            ,header = None)   
                    _df = _df.iloc[1:, :] 
                    

                else:
                    _df = pd.read_csv(datapath+'/'+in_folder,
                            engine='python',
                            encoding='utf-8')
                if dtype == "position":
                    zeros[scene][dtype] = _df[' Time'][0] 
                
                if (dtype == "position") | (dtype == "rotation"):
                    _df = _df.rename(columns={" Time": "Time"," X": "X"," Y": "Y"," Z": "Z"})
                    
                _df['Subject'] = folder

    
                unity[dtype][scene].append(_df)
 
        
    df = data['Tracker'][0][0]
    # Now split the dataframe
    mandatory_columns = ['Scene', 'Unitytime']
    df.columns = rename_duplicates(list(df.columns))
    # Identify "unit" columns
    unit_columns = [col for col in df.columns if 'unit' in col]
    # First dataframe: mandatory columns + "unit" columns
    facetracker= df[mandatory_columns + unit_columns]
    # Second dataframe: mandatory columns + all other columns not in df1
    other_columns = [col for col in df.columns if col not in mandatory_columns + unit_columns]
    eyetracker = df[mandatory_columns + other_columns]
    
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

    # Replace values in the column
    eyetracker['Scene'] = eyetracker['Scene'].map(mapping)
    facetracker['Scene'] = facetracker['Scene'].map(mapping)
    eyetracker.rename(columns={"Scene": "scene"}, inplace=True)
    facetracker.rename(columns={"Scene": "scene"}, inplace=True)
    
    
    data["Eyetracker"].append(eyetracker)
    data["Facetracker"].append(facetracker)

                

    return (names, data, unity)
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


def read_acqknowledge_with_markers(filename, sampling_rate="max", resample_method="interpolation", impute_missing=True):
    """
    Read and format a BIOPAC's AcqKnowledge file into a pandas' dataframe, including event markers.

    Parameters
    ----------
    filename : str
        Filename (with or without the extension) of a BIOPAC's AcqKnowledge file (e.g., "data.acq").
    sampling_rate : int or "max"
        Desired sampling rate in Hz. "max" uses the maximum recorded sampling rate.
    resample_method : str
        Method of resampling.
    impute_missing : bool
        Whether to impute missing values in the signal.

    Returns
    ----------
    df : DataFrame
        The AcqKnowledge file as a pandas dataframe.
    event_markers : DataFrame
        Event markers with columns ['Time (s)', 'Channel', 'Type', 'Text'].
    sampling_rate : int
        Sampling rate used in the data.

    """
    try:
        import bioread
    except ImportError:
        raise ImportError("Please install the 'bioread' module (`pip install bioread`).")

    # Check filename
    if not filename.endswith(".acq"):
        filename += ".acq"

    if not os.path.exists(filename):
        raise ValueError(f"File not found: {filename}")

    # Read the AcqKnowledge file
    file = bioread.read_file(filename)

    # Determine sampling rate
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

        # Resample signal
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
            "Time (s)": marker.sample_index / sampling_rate,
            "Channel": marker.channel_name,
            "Type": marker.type,
            "Text": marker.text
        })
    event_markers_df = pd.DataFrame(event_markers)
    event_markers_df['Sample'] = event_markers_df['Time (s)']*2000
    event_markers_df[['scene', 'marker']] = event_markers_df['Text'].str.split(',', n=2, expand=True)[[0, 1]]
    event_markers_df = event_markers_df[event_markers_df['Type'] != 'Append']
    event_markers_df.drop(columns=['Text','Channel','Type','Time (s)'], inplace=True)
    event_markers_df['scene'] = pd.to_numeric(event_markers_df['scene'])
    event_markers_df['Sample'] = event_markers_df['Sample'].apply(np.int64)
    event_markers_df.set_index('Sample', inplace=True)
    result = pd.concat([df, event_markers_df], axis=1)
    for i in range(1,8):
        startidx = result[(result['scene']==i)&(result['marker']=="Start")].index[0]
        startpidx = result[(result['scene']==i)&(result['marker']=="StartP")].index[0]
        endidx = result[(result['scene']==i)&(result['marker']=="End")].index[0]
        result.loc[startidx:endidx, 'scene'] = i
        result.loc[startidx:startpidx, 'marker'] = "Preparing"
        result.loc[startpidx:endidx, 'marker'] = "Ongoing"
    result.dropna(inplace=True)
    
    
    mapping = {
    1: "Practice",
    2: "ElevatorTest",
    3: "Elevator1",
    4: "Outside",
    5: "Hallway",
    6: "Elevator2",
    7: "Hall"
}

    # Replace values in the column
    result['scene'] = result['scene'].map(mapping)
    
    ############ Time에 sampling_rate 곱해서 frame 단위로 바꾸고, Time 기본은 남겨두고 frame을 index로 할 것.

    return df, event_markers_df, sampling_rate, result





#####################Anxiety#############################################
def clip_slider(value):
    return max(0, min(10, value)) #('Very\nCalm\n(0)','','','','','Medium\n\n(5)','','','','','Very\nAnxious\n(10)')


def Anxietyloader(filepath,name, scenes=["Elevator1", "Outside", "Hallway", "Elevator2", "Hall"]):
    # Initialize variables
    
    data = []
    current_scene = None

    # Define function to clip slider values


    # Open and process the file
    with open(filepath, 'r') as file:
        for line in file:
            # Extract time
            time_match = re.match(r'(\d+\.\d{4})', line)
            time = float(time_match.group(1)) if time_match else None

            # Check for scene changes
            if 'movie: autoDraw = True' in line:
                current_scene = scenes.pop(0) if scenes else None
            elif 'movie: autoDraw = False' in line:
                current_scene = None

            # Extract slider marker values
            slider_match = re.search(r'slider: markerPos = ([\d\.]+)', line)
            if slider_match:
                slider_value = clip_slider(float(slider_match.group(1)))
                data.append({'time': time, 'slider marker': slider_value, 'scene': current_scene, 'name': name})

    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    

    print("Processing complete")
    
    return df

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
    