from operator import index
import os
import pandas as pd 
import sqlite3
import glob
import sys 
import pathlib
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import random
import string



parser = argparse.ArgumentParser(description="python plot_performance.py -f <first step> -l <last step> ./path/to/my/sqlite/files")
parser.add_argument("-f", type=int, metavar='first step', help= "the first step")
parser.add_argument("-l", type=int, metavar='last step', help= "the last step")
parser.add_argument("-table", type=str, metavar='table', help= "the sqlite folder path")
parser.add_argument("-p", type=str, metavar='path', help= "the sqlite folder path")
parser.add_argument("-type", type=str, metavar='type', help= "sum or avg")

args = parser.parse_args()
paths = glob.glob(args.p + "*.sqlite")
df = pd.DataFrame() 
my_dict = {}
x_data = []
y_data = []
casename = re.split('-', re.split('/', paths[0])[len(re.split('/', paths[0])) - 1])
for ele in paths:
    directory = re.split('/', ele) 
    file = directory[len(directory) - 1]  
    curr_case_name = re.split('-', file)
    con =  sqlite3.connect(ele)
    cur = con.cursor()
    cur.execute(f''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{args.table}' ''')
    if cur.fetchone()[0]==1 and curr_case_name[0] == casename[0]:

        start = curr_case_name[1].index('e')
        
        end = curr_case_name[1].index('n',start+1)
        
        global_elements = curr_case_name[1][start+1:end] 
        
        
        other = (pd.read_sql_query(f"SELECT * from {args.table}", con)).iloc[args.f:args.l] # get specific range of steps from sqlite file
        print(other)
        if args.type == 'avg' :

            average = other.iloc[:,2].mean()

            y_data.append(average)
            
            x_data.append(int(global_elements))   
        elif args.type == 'sum' :
            
            summation = other.iloc[:,2].sum()
            
            y_data.append(summation)
            
            x_data.append(int(global_elements))
    
        else :
            df[global_elements] = other.iloc[:, 2]
            
    con.close()

def plot_bar_sum():
    my_dict['nelem'] = x_data
    
    my_dict['sum'] = y_data
    
    df2 = pd.DataFrame(my_dict)
    
    df2.plot(x='nelem', y='sum', style='.-').set(xlabel=f"Steps {args.f} - {args.l} ", ylabel=args.table, title=f"{casename[0]} with different mesh sizes")
    
    plt.show()
    
def plot_bar_avg():
    my_dict['nelem'] = x_data
    
    my_dict['avg'] = y_data
    
    df2 = pd.DataFrame(my_dict)
    
    df2.plot(x='nelem', y='avg', style='.-').set(xlabel=f"Steps {args.f} - {args.l} ", ylabel=args.table, title=f"{casename[0]} with different mesh sizes")
    
    plt.show()
    
def plot_steps():
    ax = df.plot.bar()
    
    ax.set(xlabel=f"Number of Elements", ylabel=args.table, title=f"{casename[0]} with different mesh sizes, Steps {args.f} - {args.l}")
    
    plt.show()
    
def box_plot():
    fig, ax = plt.subplots()
    
    ax.set_xticklabels(df.columns, rotation=0)
    
    ax.set(xlabel=f"Steps {args.f} - {args.l} ", ylabel=args.table, title=f"{casename[0]} with different mesh sizes")
    
    for n, col in enumerate(df.columns):
        
        ax.boxplot(df[col], positions=[n+1], notch=True)
    
    plt.show()
    
    
def get_x_tick_labels(df, grouped_by):
    tmp = df.groupby([grouped_by]).size()
    return ["{0}: {1}".format(k,v) for k, v in tmp.to_dict().items()]

def series_values_as_dict(series_object):
    tmp = series_object.to_dict().values()
    return [y for y in tmp][0]

def add_values(bp, ax):
    for element in ['whiskers', 'medians', 'caps']:
        for line in bp[element]:
            
            
            (x_l, y),(x_r, _) = line.get_xydata()

            if not np.isnan(y): 
                x_line_center = x_l + (x_r - x_l)/2
                y_line_center = y  # Since it's a line and it's horisontal
                # overlay the value:  on the line, from center to right
                ax.text(x_line_center, y_line_center, # Position
                        '%.3f' % y, # Value (3f = 3 decimal float)
                        verticalalignment='center', # Centered vertically with line 
                        fontsize=16, backgroundcolor="white")
if args.type == 'avg':
    plot_bar_avg()
elif args.type == 'sum' :
    plot_bar_sum()
else:
    box_plot()



