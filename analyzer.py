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
        print(args.type)
        if args.type == 'avg' :
            

            average = other.mean(axis=1)
            
            df[global_elements] = average
        
        elif args.type == 'sum' :
            

            print(other['value'].sum())
            #summation = other.sum(axis=1)  
            
            summation = other['value'].sum()
            df[global_elements] = summation
            #df.insert(0, 'col_name', summation)
            
            
            print(df.head())
            print(df[global_elements])
            print("yo")
        else :
            df[global_elements] = other.iloc[:, 2]
            
    con.close()

for col_name in df.columns: 
    print(col_name)
print(df.columns.values[1])

def plot_steps():
    ax = df.plot.bar()
    ax.set(xlabel=f"Steps {args.f} - {args.l} ", ylabel=args.table, title=f"{casename[0]} with different mesh sizes")
    plt.show()
def box_plot():
    fig, ax = plt.subplots()
    ax.set_xticklabels(df.columns, rotation=0)
    ax.set(xlabel=f"Steps {args.f} - {args.l} ", ylabel=args.table, title=f"{casename[0]} with different mesh sizes")
    for n, col in enumerate(df.columns):
        ax.boxplot(df[col], positions=[n+1], notch=True)
        ax.annotate('local max', xy = (col, df[col].median()), xytext =(col, df[col].median() + 20), arrowprops = dict(facecolor ='green', shrink = 0.05),)
   
    plt.show()
    


box_plot()
#plot_steps()
# df.plot.box()


# import datetime
# import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt

# def fig1():
#   fig = plt.figure(constrained_layout=True)
  
  
#   ax = plt.axes()
#   ax.set_facecolor('yellowgreen')

#   col = 'darkolivegreen'
#   ax.spines['bottom'].set_color(col)    #plot border
#   ax.spines['top'].set_color(col) 
#   ax.spines['right'].set_color(col)
#   ax.spines['left'].set_color(col)
#   ax.tick_params(axis='x', colors=col)  #ticks
#   ax.tick_params(axis='y', colors=col)
#   ax.yaxis.label.set_color(col)       #axis label
#   ax.xaxis.label.set_color(col)       
#   ax.title.set_color(col)             #title

#   plt.plot(range(7), [3, 1, 4, 1, 5, 0, 2], 'w-o')
#   plt.show()  
  
# fig1()  


