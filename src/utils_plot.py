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



def draw_corr_pval(df, function=stats.spearmanr, nan="omit"):
    K = len(df.columns)
    Rvalue = np.empty((K, K), dtype=float)
    pvalue = np.empty((K, K), dtype=float)
    for i, col1 in enumerate(list(df.columns)):
        for j, col2 in enumerate(list(df.columns)):
            if i > j:
                continue
            else:
                r, p = function(df[col1], df[col2], nan_policy=nan)
            Rvalue[i, j] = r
            Rvalue[j, i] = r
            pvalue[i, j] = p
            pvalue[j, i] = p
    pvalue = pvalue + np.identity(K)  # 자명한 부분은 1을 넣어버려 별을 없애 깔끔히 만들쟈!!!
    Rvalue = pd.DataFrame(Rvalue, index=df.columns, columns=df.columns)
    pvalue = pd.DataFrame(pvalue, index=df.columns, columns=df.columns)
    p = pvalue.applymap(
        lambda x: "\n" + "".join(["*" for t in [0.005, 0.01, 0.05] if x <= t])
    )
    annot = Rvalue.round(3).astype(str) + p
    plt.figure(figsize=(12, 12))
    sns.heatmap(Rvalue, annot=annot, fmt="", cmap='RdBu', vmin=-1, vmax=1)
    return Rvalue, pvalue, annot


def cutplot(
    x,
    y,
    df,
    hue: str = None,
    style: str = None,
    name="Test",
    palette="Set2",
    cuttime_v=300,
    cuttime_h: float = None,
    color_v="lightgray",
    color_h="lightgray",
    style_v="--",
    style_h="--",
    ylim: list = [0, 5.0],
    filedir="graphs",
    close_img=True,
    title_size="xx-large",
    label_size="xx-large",
):
    """Seaborn plotting for cut data. Result will be saved in .png formet

    Args:
        x (str): x axis column name
        y (str): y axis column name
        hue (str): Variable for hue arguement in seaborn plotting
        style (str): Variable for style arguement in seaborn plotting
        df (df): dataframe of the plotting data
        name (str): plot name
        palette (str, optional): Variable for palette arguement in seaborn plotting. Defaults to 'Set2'.
        cuttime_v (int, optional): axvline number. Defaults 300.
        cuttime_h (int, optional): axhline number. Defaults 300.
        color_v (int, optional): axvline color. Defaults 'lightgray'.
        color_h (int, optional): axhline color. Defaults 'lightgray'.
        style_v (int, optional): axvline style. Defaults '--'.
        style_h (int, optional): axhline style. Defaults '--'.
        ylim (list, optional): ylim number. Defaults 300.
        filedir (str, optional): file path to save. if None, not save the file. Defaults to 'graphs/'.
        close_img (bool, optional): If True, do not show image on cell. Defaults to 'graphs/'.
        title_size (str, optional): title fontsize. Defaults to "xx-large".
        label_size (str, optional): lable fontsize. Defaults to "xx-large".
        
    """
    sns.lineplot(x=x, y=y, hue=hue, style=style, palette=palette, data=df)
    if cuttime_v != None:
        plt.axvline(cuttime_v, color=color_v, linestyle=style_v)
    if cuttime_h != None:
        plt.axvline(cuttime_h, color=color_h, linestyle=style_h)
    plt.title(name, fontsize=title_size)
    plt.ylim(ylim)
    plt.xlabel(x, fontsize=label_size)
    plt.ylabel(y, fontsize=label_size)
    plt.legend()
    if filedir != None:
        plt.savefig(os.path.join(filedir, f"{name}.png"))
    if close_img:
        plt.close()
        
        
def drawheatmap_ttest(df: pd.DataFrame, name: str = 'Test', pivot: str = 'p-value', pallate: list = [130,120], pg: bool = True, save: bool = True, notshow: bool = True, fontsize: list = ['xx-large','x-large'], cmap="YlGnBu"):
    """Draw heatmap with t_val df function result.

    Args:
        df (pd.DataFrame): t_val df function. Containing tvalue and pvalue and other condition labels.
        name (str, optional): Figure Title name. File will be saved based on this name. Defaults to 'Test'.
        pallate (list, optional): Pallate number list. Defaults to [130,120].
        save (bool, optional): Whether you want to save result figure. Defaults to True.
        notshow (bool, optional): Whether you want to see the result figure in cell. Defaults to True.
    """
    if pg:
        dof = df['dof']
        col1, col2 = df.iloc[:,-2:].columns
        print(pd.concat([dof,df.iloc[:,-2:]],axis=1))
        pval = 'p-val'
    else:
        labels = list(df.columns)
        labels.remove('t-value')
        labels.remove('p-value')
        col1, col2 = labels
        pval = 'p-value'
        
    
    _df = df.pivot(col1,col2, pivot)
    print(_df)
    p = df.pivot(col1,col2, pval).applymap(lambda x: "\n" + "".join(["*" for t in [0.005, 0.01, 0.05] if x <= t]))
    # d = _df.applymap(lambda x: "\n" + "".join(["(-)" if x < 0 else ""]))
    _df = abs(_df)
    annot = _df.round(3).astype(str) + p
    # cmap = sns.diverging_palette(pallate[0],pallate[1], as_cmap=True)
    if pg:
        sns.heatmap(_df, annot=annot,annot_kws={"size": 20}, fmt="",cmap=cmap, vmin=0, vmax=stats.t.ppf(0.995, dof.mean()))
    else:
        sns.heatmap(_df, annot=annot,annot_kws={"size": 20}, fmt="",cmap=cmap)
    plt.title(name, fontsize=fontsize[0])
    plt.ylabel(col1,fontsize=fontsize[0])
    plt.xlabel(col2,fontsize=fontsize[0])
    plt.tick_params(labelsize=fontsize[1])
    if save:
        plt.savefig(f'graphs/{name}.png')
    if notshow:
        plt.close()
    
        