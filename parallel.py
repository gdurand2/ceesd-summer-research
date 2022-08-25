from curses import color_content
from operator import index
import os
from tkinter import font
from turtle import color
from unittest import case
import colorama
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

import pandas as pd

parser = argparse.ArgumentParser(description="python plot_performance.py -f <first step> -l <last step> ./path/to/my/sqlite/files")
parser.add_argument("-f", type=int, metavar='first step', help= "the first step")
parser.add_argument("-l", type=int, metavar='last step', help= "the last step")
parser.add_argument("-table", type=str, metavar='table', help= "the sqlite folder path")
parser.add_argument("-p", type=str, metavar='path', help= "the sqlite folder path")
parser.add_argument("-type", type=str, metavar='type', help= "sum or avg")

parser.add_argument("-dash", type=bool, metavar='dash', help= "true or false")
parser.add_argument("-date", type=int, metavar='date', help= "date ")

args = parser.parse_args()
paths = glob.glob(args.p + "*.sqlite")
df = pd.DataFrame() 
my_dict = {}
steps = {}
x_data = []
y_data = []
t_step_data = []

t_step_data_sum = [] #new 
t_step_data_avg = [] #new
element_counter = {}
t_init_data = []
memory_usage_python = []
compile_time_data = []

casename = re.split('-', re.split('/', paths[0])[len(re.split('/', paths[0])) - 1])

for ele in paths:
    directory = re.split('/', ele) 
    file = directory[len(directory) - 1]  
    curr_case_name = re.split('-', file)
    con =  sqlite3.connect(ele)
    cur = con.cursor()
   
    date = int(curr_case_name[len(curr_case_name) - 2])
    time_stamp = int(curr_case_name[len(curr_case_name) - 1].replace('.sqlite', ''))
    
    if args.table is not None and args.date == date:
        cur.execute(f''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{args.table}' ''')
        if cur.fetchone()[0]==1 and curr_case_name[0] == casename[0]:
            other = (pd.read_sql_query(f"SELECT * from {args.table}", con)).iloc[args.f:args.l] # get specific range of steps from sqlite file
            summation = other.iloc[:,2].sum()  
            if(time_stamp in my_dict.keys()):
                my_dict[time_stamp].extend([summation])
            else:
                my_dict[time_stamp] = [summation]
    con.close()
    
def sort_by_values_len(dict):
    print(dict)
    dict_len= {key: len(value) for key, value in dict.items()}
    print(dict_len)
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=False)
    sorted_dict = [{item[0]: dict[item [0]]} for item in sorted_key_list]
    print(sorted_dict)
    rank_max = [{len(dict[item[0]]): max(dict[item [0]])} for item in sorted_key_list]
    result = {}
    for d in rank_max:
        result.update(d)
    
    return result
        
        
def create_plot():
    sorted_my_dict = sort_by_values_len(my_dict)
    print(sorted_my_dict)
    t_dict = {}
    x_values = list(sorted_my_dict.keys())
    y_values = list(sorted_my_dict.values())
    t_dict['rank'] = x_values
    t_dict['t_step_max'] = y_values

    df = pd.DataFrame(t_dict)
    fig, ax = plt.subplots()
    df.plot(x='rank',
             y='t_step_max',
             style='.-',
             grid=True,
             marker='o',
             ms=4,
             markerfacecolor='w',
             ax = ax).set(xlabel=f"Number of Ranks",
                          ylabel="t_step max",
                          title=f"{casename[0]} time taken cross ranks, Steps {args.f} - {args.l}")
    plt.show()
    print(df.head())     
        
        
create_plot()        
