import os
from collections import defaultdict
import re
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from itertools import product
import pickle

def createDir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def time2frame(timepoint, textformet=True, frame = 30):
    """coughing data is textformet = True. Text formet: min:sec:sec"""
    if textformet == True:
        minute, second = map(float, timepoint.split(":"))
        total_second = second + minute * 60
    else:
        total_second = timepoint
    return round(total_second * frame)


def metadata2frame(metadata: dict):
    for meta, value in metadata.items():
        if isinstance(value, str):
            metadata[meta] = time2frame(value)
    return metadata


def coughxlsx2df(xlsxname: list):
    """Change cough_count_manual xlsx files to dataframe types
    coughing dictionary will contain two divided divided data-coughing df and metadata
    coughing df contains 1) type of coughing(full(f), soundonly(s), actiononly(a))
    2) Start and End timepoint of each coughing event (str type)
    metadata contains 1) Total video lenght 2) Goal On Sight 3) and others, in dict
    Args:
        xlsxname (list): names of the xlsx files in cough folder.
    """
    meta_columns = [
        "Total_video_length",
        "Total_count_full_insight",
        "Total_count_soundonly",
        "Total_count_actiononly",
        "Goal_Start toward moving",
        "Goal_On sight",
    ]

    seq_columns = [
        "Type(full_insight(f), soundonly(s), actiononly(a))",
        "Start_timepoint",
        "End_timepoint",
    ]
    for filename in xlsxname:
        coughing_excel = pd.ExcelFile(filename)
        coughing = defaultdict(dict)

        for idx in range(len(coughing_excel.sheet_names)):
            tmp_df = pd.read_excel(filename, sheet_name=idx)
            tmp_df = tmp_df.drop(["Unnamed: 3"], axis=1)

            metadata = metadata2frame(tmp_df[meta_columns].loc[0].to_dict())
            name = coughing_excel.sheet_names[idx]
            if name == "template":
                continue
            else:
                name, mask, place = name.split("_")

            coughing[f"{mask}_{place}"][name] = (tmp_df[seq_columns], metadata)
    return coughing


def processed2df(filenames: list):
    """Change processed data which in formet of csv to dataframe type.

    Args:
        filename (list): names of csv files in file folder.

    Returns:
        _type_: dictionaries of the file data in {mask on/off}_{place}[name] formet
    """

    file = defaultdict(dict)
    for filename in filenames:
        file_csv = pd.read_csv(filename, index_col =[0])
        _, _, place, name, mask, _ = re.split("_|.csv", filename)
        file[f"{mask}_{place}"][name] = file_csv

    return file


def cut2case(
    targetdata: dict,
    anxietydata: dict,
    emptylist: list,
    typename: str,
    pointname: str,
    interval=30,
    metadata=True,
    name=None,
    bubble=False
):
    """cut df into each event cases. All data should have same index(ex)fps).
    Args:
        targetdata (dict): target data of event. For example, coughing or bubble data. You need to put dict['mask_place'] formet in target and anxiety data
        anxietydata (dict): anxiety data. This will be divided and appended to the result data.
        emptylist (list): List which will contain cuts of the datas
        typename (str): column name of the type.
        pointname (str): column name of the point. For example, if onset targeted, 'Start_fps' will be the name.
        interval (int, optional): interval frame. Defaults to 30.
        metadata (bool): If True, there are extra metadata. so select first index data to process
        name: If you want to process data directly with dataframe, please assign participant name.
        bubble: If true, you are processing personal distance(bubble) data. Add distance info at the list
    """
    if metadata:
        for participants in targetdata:
            df = targetdata[participants][0]
            for index in range(len(df)):
                dtype = df[typename][index]
                point = df[pointname][index]

                _df = anxietydata[participants][point - interval : point + interval + 1]
                _df = _df.reset_index(drop=True)
                emptylist.append((_df, dtype, participants))\
                    
        
    else:
        df = targetdata
        if bubble:
            b_list = df['Bubble'].unique()
            for dist in b_list:
                df_temp = df[df['Bubble']==dist]
                for index in range(len(df_temp)):
                    dtype = df_temp[typename][index]
                    point = df_temp[pointname][index]
                    if (point - interval) < 0:
                        start = 0
                    else:
                        start = point - interval
                    _df = anxietydata[start : point + interval + 1]
                    _df = _df.reset_index(drop=True)
                    emptylist.append((_df, dtype, name, dist))
            
        else:
        
            for index in range(len(df)):
                dtype = df[typename][index]
                point = df[pointname][index]
                if bubble:
                    dist = df['Bubble'][index]
                    print(dist)
                if (point - interval) < 0:
                    start = 0
                else:
                    start = point - interval
                _df = anxietydata[start : point + interval + 1]
                _df = _df.reset_index(drop=True)
                emptylist.append((_df, dtype, name))
                

    return emptylist


def cut2single(
    targetdata: dict,
    anxietydata: dict,
    emptylist1: list,
    emptylist2: list,
    pointname: str,
    interval=30,
):
    """cut df with single meta data. All data should have same index(ex)fps). ex) Goal on sight
    Args:
        targetdata (dict): target data of event. For example, coughing or bubble data. You need to put dict['mask_place'] formet in target and anxiety data
        anxietydata (dict): anxiety data. This will be divided and appended to the result data.
        emptylist (list): List which will contain cuts of the datas
        pointname (str): key name for target goal
        interval (int, optional): interval frame. Defaults to 30.
    """
    for participants in targetdata:
        point = targetdata[participants][1][pointname]
        # Around the goal
        _df1 = anxietydata[participants][point - interval : point + interval + 1]
        _df1 = _df1.reset_index(drop=True)
        # Before goal
        _df2 = anxietydata[participants][: point + 1]
        _df2 = _df2.reset_index(drop=True)

        # After goal
        _df3 = anxietydata[participants][point:]
        _df3 = _df3.reset_index(drop=True)

        # To see flow of anxiety around goal reaching timing
        emptylist1.append(_df1)
        # Anxiety Before goal mean value, Anxiety After goal mean value
        emptylist2.append([_df2.mean()[0], _df3.mean()[0]])

    return emptylist1, emptylist2




def _make_query(col: str, cond: str):
    """Create query formet. used inside make_queries
    returns str"""
    return f"{col} == '{cond}'"

def make_queries(queries: List[Tuple,]):
    """Create a list of queries to use for query
    returns str"""
    return " & ".join([_make_query(col, cond) for col, cond in queries])

def query(df: pd.DataFrame, queries: List[Tuple,], val_col: str):
    """do search with queries to get specific conditioned anxiety groups
    returns pd.Series"""
    return df.query(make_queries(queries))[val_col]

def query_w(df: pd.DataFrame, queries: List[Tuple,], val_col: str, within_col: str = 'Participants'):
    """do search with queries to get specific conditioned anxiety groups
    returns pd.Series. Withing Case"""
    return df.query(make_queries(queries)).groupby(within_col)[val_col].mean()
 
 
def getqueries(df: pd.DataFrame, omit_col: list = ['Anxiety','FPS','Participants'], target_col: str = 'Case'):
    """To divide target and other columns, so we can get proper sets of queries to compare.
    returns List(Tuple)"""
    queries_f = []
    queries_t = []
    
    for col in df:
        if col == target_col:
            columns = []
            for item in df[col].unique():
                columns.append((col,item))
            queries_t = columns
        if  (col not in omit_col) & (col != target_col):
            columns = []
            for item in df[col].unique():
                columns.append((col,item))
            queries_f.append(columns)

    return queries_f, queries_t

def tval_df(df, target_col: str, omit_col: list = ['Anxiety','FPS','Participants'], 
            val_col: str = 'Anxiety',function=stats.ttest_ind, pg: bool = True ,within: bool = False):
    """Get dataframe of t-value and p-value based on ttest of the function. We can use it for heatmap drawing

    Args:
        df (pd.DataFrame): dataframe to perform ttest
        target_col (str): target column name. We will divide dataframe based on this column
        omit_col (list, optional): Omit unnecessary columns. We will not going to compare with these labels. Defaults to ['Anxiety','FPS','Participants'].
        val_col (str, optional): We will perform t-test based on this label value. Defaults to 'Anxiety'.
        function (_type_, optional): ttest function. Defaults to stats.ttest_ind.
        within (bool, optional): Within test true/false. Defaults to stats.ttest_ind
    """
    queries_f, queries_t = getqueries(df, omit_col=omit_col, target_col =target_col)
    fixed = list(product(*queries_f))
    target = queries_t
    heatmap_df = pd.DataFrame()
    _df = pd.DataFrame()
  
    for idx, items in enumerate(fixed):
        queries_ = []
        name_dict = {}
        for condition in target:
            queries_.append(items + ((condition),))

        if pg:
                case1 = query(df,queries_[0], val_col)
                case2 = query(df,queries_[1], val_col) 
        else:
            if within:
                case1 = query_w(df,queries_[0], val_col)
                case2 = query_w(df,queries_[1], val_col)
            else: 
                case1 = query(df,queries_[0], val_col)
                case2 = query(df,queries_[1], val_col)
        print(f'X:{queries_[0]}, Y:{queries_[1]}')
        if pg:
            if within:
                heatmap_df = pd.concat([heatmap_df, function(case1, case2, paired=True)], axis=0)
                # heatmap_df = heatmap_df.reset_index()

            else:
                heatmap_df = pd.concat([heatmap_df, function(case1, case2).reset_index()], axis=0)
                
            for col, cond in items:
                name_dict[col] = cond
            _df = _df.append(name_dict,ignore_index=True)
            _df = _df.set_index(heatmap_df.index)
            
                
        else: 
            tval, pval = function(case1,case2)
            name_dict['t-value'] = tval
            name_dict['p-value'] = pval
            for col, cond in items:
                name_dict[col] = cond
                
            heatmap_df = heatmap_df.append(name_dict,ignore_index=True)
            
            

    heatmap_df = pd.concat([heatmap_df,_df],axis=1)
       
    
    return(heatmap_df)


def bubbles2e(df):
    whole_df = pd.DataFrame()
    tmp_list = []
    for distance in df.columns:
        _df = pd.DataFrame()
        for idx in range(len(df)):
            if idx == 0:
                continue
            else:
                current = df[distance][idx] 
                previous = df[distance][idx-1] 
                if current > previous:
                    startframe = idx
                    ppl_st = current
                    if previous != 0:
                        endframe = idx
                        ppl_ed = previous 
                        _dict = {'Start_Frame':startframe, 'End_Frame':endframe,'Agent_Number':ppl_ed}
                        _df = _df.append(_dict, ignore_index=True)
                        # print('end_val: ', previous, idx)
                        # print('_______________')

                    # print('start_val: ',current, idx)
                if previous > current:
                    endframe = idx
                    ppl_ed = previous
                    # print('end_val: ',previous, idx)
                    # print('_______________')
                    if current != 0:
                        startframe = idx
                        ppl_st = current
                        # print('start_val: ', current, idx) 
                    _dict = {'Start_Frame':startframe, 'End_Frame':endframe,'Agent_Number':ppl_ed}
                    _df = _df.append(_dict, ignore_index=True)
        _df['Bubble'] = distance.split()[0]  
        
        tmp_list.append(_df)     

    whole_df = pd.concat(tmp_list)   
    return whole_df


def createFolder(directory: str):
    """Create Folder and check whether the folder name already occupied or not

    Args:
        directory (str): directory name
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def data2csv(data,names: list, filename: str,directory: str = './data/processed/',single: bool = False):
    """Save data as .csv formet. data should be compatable format with .csv
    List of dataframe is recommended.

    Args:
        data (_type_): data to save. List of dataframe is recommended
        names (list): names of participants. This added because it also need to be iterated if it is not the single form
        filename (str): file name to save. This should 'not' inclue "/". Just names. ex) 'Anxiety_out_'
        directory (str, optional): Directory to save. Please Change it properly. Defaults to './data/processed/'.
        single (bool, optional): Just for if you want to save single dataframe as .csv. If so, put it True. Defaults to False.
    """
    createFolder(directory)
    if single == False:
        for idx in range(len(data)):
            data[idx].to_csv(directory + "/" + filename + names[idx] +'.csv', sep = ',', na_rep = 'NaN', index=True)
    else:
        data.to_csv(directory + "/" + filename + '.csv', sep = ',', na_rep = 'NaN', index=True)

def save2pkl(data, filename: str):
    """save data in a .pkl file. 
    If you want to use dataframe, list or dict again at other file, this is recommended.
    Especially if it is single file.

    Args:
        data (_type_): data
        filename (str): This should be full directory + filename.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        

def openpkl(directory: str):
    """open .pkl file

    Args:
        directory (str): This should be full directory including filename
    """
    with open(directory, 'rb') as handle: #Get bubble-agent data
        variable = pickle.load(handle)
        
    return variable
