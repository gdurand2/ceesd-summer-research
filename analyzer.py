from operator import index
import os
from tkinter import font
from turtle import color
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
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
import random
import base64
import dash_daq as daq


parser = argparse.ArgumentParser(description="python plot_performance.py -f <first step> -l <last step> ./path/to/my/sqlite/files")
parser.add_argument("-f", type=int, metavar='first step', help= "the first step")
parser.add_argument("-l", type=int, metavar='last step', help= "the last step")
parser.add_argument("-table", type=str, metavar='table', help= "the sqlite folder path")
parser.add_argument("-p", type=str, metavar='path', help= "the sqlite folder path")
parser.add_argument("-type", type=str, metavar='type', help= "sum or avg")
parser.add_argument("-annotate", type=bool, metavar='annotate', help= "true or false")

args = parser.parse_args()
paths = glob.glob(args.p + "*.sqlite")
df = pd.DataFrame() 
my_dict = {}
steps = {}
x_data = []
y_data = []
t_step_data = []
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
    start = curr_case_name[1].index('e')         
    end = curr_case_name[1].index('n',start+1) 
    global_elements = curr_case_name[1][start+1:end] 
    
    if args.table is not None:
        cur.execute(f''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{args.table}' ''')
        if cur.fetchone()[0]==1 and curr_case_name[0] == casename[0]:
            other = (pd.read_sql_query(f"SELECT * from {args.table}", con)).iloc[args.f:args.l] # get specific range of steps from sqlite file
            
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
    else:
        if args.type == 'sum' :
            x_data.append(int(global_elements))
            t_step_sum = (pd.read_sql_query(f"SELECT * from t_step", con).iloc[args.f:args.l,2]).sum() 
            t_step_data.append(t_step_sum)
            t_init = pd.read_sql_query(f"SELECT * from t_init", con).iloc[:,2]
            t_init_data.append(float(t_init))
            memory_usage_python_max = (pd.read_sql_query(f"SELECT * from memory_usage_python", con).iloc[args.f:args.l,2]).max()
            memory_usage_python.append(memory_usage_python_max)
        else:
            steps[int(global_elements)] = (pd.read_sql_query(f"SELECT * from t_step", con).iloc[args.f:args.l,2])
            x_data.append(int(global_elements))
            t_step_avg = (pd.read_sql_query(f"SELECT * from t_step", con).iloc[args.f:args.l,2]).mean()
            t_step_data.append(t_step_avg)
            t_init = pd.read_sql_query(f"SELECT * from t_init", con).iloc[:,2]
            t_init_data.append(float(t_init))
            memory_usage_python_max = (pd.read_sql_query(f"SELECT * from memory_usage_python", con).iloc[args.f:args.l,2]).max()
            memory_usage_python.append(memory_usage_python_max)
            compile_time = (pd.read_sql_query(f"SELECT * from t_step", con).iloc[0:1,2])
            print(compile_time)
            compile_time_data.append(float(compile_time))
            
    
    con.close()
    
csfont = {'fontname':'DejaVu Sans'}   
df_steps = pd.DataFrame(steps)
df_steps = df_steps.reindex(sorted(df_steps.columns), axis=1)
df_steps.index.name = "steps"
print(len(df_steps.columns.values))
print(df_steps)
def plot_all():
    fig, axes = plt.subplots(nrows=4, ncols=1)
    my_dict['nelem'] = x_data
    my_dict['t_step'] = t_step_data
    my_dict['t_init'] = t_init_data
    my_dict['memory max'] = memory_usage_python
    my_dict['compile time'] = compile_time_data
    df2 = pd.DataFrame(my_dict)
    df2 = df2.sort_values(by='nelem')

    fig, (ax1,ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    ax5 = fig.add_subplot(111, zorder=-1)
    for _, spine in ax5.spines.items():
        spine.set_visible(False)
    ax5.tick_params(labelleft=False, labelbottom=False, left=False, right=False )
    ax5.get_shared_x_axes().join(ax5,ax1)
    ax5.get_shared_x_axes().join(ax5,ax2)
    ax5.get_shared_x_axes().join(ax5,ax3)
    ax5.grid(axis="x")

    line1 = ax1.plot(df2['nelem'],
                     df2['t_step'],
                     markerfacecolor='w',
                     ms=4,
                     marker='o',
                     color='orange',
                     label="t_step")
    ax1.set_ylabel("time (s)", csfont, fontsize=8)
    line1 = ax2.plot(df2['nelem'],
                     df2['t_init'],
                     ms=4,
                     markerfacecolor='w',
                     marker='o',
                     color='green',
                     label="t_init")
    ax2.set_ylabel("time (s)", csfont, fontsize=8)
    line1 = ax3.plot(df2['nelem'],
                     df2['memory max'],
                     ms=4,
                     markerfacecolor='w',
                     marker='o',
                     color='blue',
                     label="memory max")
    ax3.set_ylabel("time (s)", csfont, fontsize=8)
    line1 = ax4.plot(df2['nelem'],
                     df2['compile time'],
                     ms=4,
                     markerfacecolor='w',
                     marker='o',
                     color='purple',
                     label="compile time")
    ax4.set_ylabel("time (s)", csfont, fontsize=8)
    fig.legend(loc='upper left', labelspacing=.01, borderaxespad=.1, fontsize='small', markerscale=.6)
    fig.align_ylabels()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.title(f"{casename[0]} with different mesh sizes between steps {args.f} - {args.l}", csfont, fontsize="small")
    plt.xlabel(f"Number of Elements", labelpad=20) 
    plt.show()
    
    
def plot_bar_sum():
    my_dict['nelem'] = x_data
    my_dict['sum'] = y_data 
    df2 = pd.DataFrame(my_dict)
    df2 = df2.sort_values(by='nelem')
    fig, ax = plt.subplots()
    df2.plot(x='nelem',
             y='sum',
             style='.-',
             grid=True,
             marker='o',
             ms=4,
             markerfacecolor='w',
             ax = ax).set(xlabel=f"Steps {args.f} - {args.l} ",
                          ylabel=args.table,
                          title=f"{casename[0]} with different mesh sizes")
    if args.annotate:
        for x,y in zip(x_data,y_data):
            label = x
            plt.annotate(label,
                        (x,y),
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8) 
    plt.show()
    
def plot_bar_avg():
    my_dict['nelem'] = x_data
    my_dict['avg'] = y_data
    df2 = pd.DataFrame(my_dict)
    df2 = df2.sort_values(by='nelem')
    fig, ax = plt.subplots()
    df2.plot(x='nelem',
             y='avg',
             style='.-',
             grid=True,
             marker='o',
             ms=4,
             markerfacecolor='w',
             ax = ax).set(xlabel=f"Steps {args.f} - {args.l} ",
                          ylabel=args.table,
                          title=f"{casename[0]} with different mesh sizes")
    if args.annotate:
        print("whatup")
        for x,y in zip(x_data,y_data):
            label = x
            plt.annotate(label,
                        (x,y),
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8) 
            
    plt.show()
    
def plot_steps():
    ax = df.plot.bar()
    ax.set(xlabel=f"Number of Elements",
           ylabel=args.table,
           title=f"{casename[0]} with different mesh sizes, Steps {args.f} - {args.l}")
    plt.show()
    
def box_plot():
    fig, ax = plt.subplots()
    ax.set_xticklabels(df.columns, rotation=0)
    ax.set(xlabel=f"Steps {args.f} - {args.l} ",
           ylabel=args.table,
           title=f"{casename[0]} with different mesh sizes")
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


##############################################################
        #DATA MANIPULATION (MODEL)
##############################################################
df= pd.read_csv("/Users/georgesdurand/testing-3/dash_dataquest/top500_clean.csv")
df['userscore'] = df['userscore'].astype(float)
df['metascore'] = df['metascore'].astype(float)
df['releasedate']=pd.to_datetime(df['releasedate'], format='%b %d, %Y')
df['year']=df["releasedate"].dt.year
df['decade']=(df["year"]//10)*10
#cleaning Genre
df['genre'] = df['genre'].str.strip()
df['genre'] = df['genre'].str.replace("/", ",")
df['genre'] = df['genre'].str.split(",")
#year trend
df_linechart= df.groupby('year')        .agg({'album':'size', 'metascore':'mean', 'userscore':'mean'})        .sort_values(['year'], ascending=[True]).reset_index()
df_linechart.userscore=df_linechart.userscore*10
#table
df_table= df.groupby('artist').agg({'album':'size', 'metascore':'sum', 'userscore':'sum'})
#genrebubble
df2=(df['genre'].apply(lambda x: pd.Series(x)) .stack().reset_index(level=1, drop=True).to_frame('genre').join(df[['year', 'decade', 'userscore', 'metascore']], how='left') )
df_bubble=  df2.groupby('genre')        .agg({'year':'size', 'metascore':'mean', 'userscore':'mean'})             .sort_values(['year'], ascending=[False]).reset_index().head(15)
df2_decade=df2.groupby(['genre', 'decade']).agg({'year':'size'}) .sort_values(['decade'], ascending=[False]).reset_index()




##############################################################
        #DATA LAYOUT (VIEW)
##############################################################

#gererate table
def generate_table(dataframe, max_rows=10):
    '''Given dataframe, return template generated using Dash components
    '''
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'middle'}
    )

#generate bar chart
# def  bar(results): 
#     gen =results["points"][0]["text"]
#     figure = go.Figure(
#         data=[
#             go.Bar(x=df2_decade[df2_decade.genre==gen].decade, y=df2_decade[df2_decade.genre==gen].year)
#         ],
#         layout=go.Layout(
#             title="Decade populatrity of " + gen
#         )
#     )
#     return figure

def  bar(results): 
    print(results)
    elements =results["points"][0]['x']
    print(elements)
    print(df_steps[elements])
    print(df_steps.index)
    figure = go.Figure(
        data=[
            go.Bar(x=df_steps.index, y=df_steps[elements], marker=dict(color=df_steps[elements], coloraxis="coloraxis"))
        ],
        layout=go.Layout(
            title="Decade populatrity of ", 
            coloraxis=dict(colorscale='Bluered_r'),
            showlegend=False
        )
    )
    figure.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)
    return figure

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
def subplots():
    my_dict['nelem'] = x_data
    my_dict['t_step'] = t_step_data
    my_dict['t_init'] = t_init_data
    my_dict['memory max'] = memory_usage_python
    my_dict['compile time'] = compile_time_data
    df3 = pd.DataFrame(my_dict)
    df3 = df3.sort_values(by='nelem')
    fig = make_subplots(rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    subplot_titles=(f"{casename[0]} with different mesh sizes",),
                    x_title="Number of Elements",
                    y_title=("Time (s)"))
    fig.add_trace(go.Scatter(x=df3['nelem'],
                             y=df3['t_step'],
                             name = 't_step',
                             text = df3['t_step']),
                  row=4, col=1)
    
    fig.add_trace(go.Scatter(x=df3['nelem'],
                             y=df3['t_init'],
                             name = 't_init'),
                row=3, col=1)

    fig.add_trace(go.Scatter(x=df3['nelem'],
                             y=df3['memory max'],
                             name = 'memory max'),
                row=2, col=1)

    fig.add_trace(go.Scatter(x=df3['nelem'],
                             y=df3['compile time'],
                             name = 'compile time'),
                row=1, col=1)
    
    fig.update_yaxes(title_text="yaxis 1 title",
                     row=4, col=1)
    fig.update_yaxes(title_text="yaxis 2 title",
                     row=3, col=1)
    fig.update_yaxes(title_text="yaxis 3 title",
                     row=2, col=1)
    fig.update_yaxes(title_text="yaxis 4 title",
                     row=1, col=1)
    fig.update_layout(title="Plot Title",
    xaxis_title="X Axis Title",
    yaxis_title="Y Axis Title",
    legend_title="Legend Title",
    # coloraxis=dict(colorscale='Bluered_r'),
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ))
    return fig

def plot_line():
    steps_dict = {}
    steps_dict['nelem'] = x_data
    steps_dict['sum'] = t_init_data
    df2 = pd.DataFrame(steps_dict)
    print(df2)
    df2 = df2.sort_values(by='nelem') 
    
    fig = px.line(df2, x='nelem', y="sum", title="t_init", markers=True) 
    
    fig = fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell",
    legend=dict(
        title=None, orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
    )
    )
    
    return fig
    
       
# Set up Dashboard and create layout
app = dash.Dash()

# Bootstrap CSS.
app.css.append_css({
    "external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
})
# Bootstrap Javascript.
app.scripts.append_script({
    "external_url": "https://code.jquery.com/jquery-3.2.1.slim.min.js"
})
app.scripts.append_script({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
})
app.scripts.append_script({
    "external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
})

#define app layout

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image, ImageEnhance
from PIL import Image
import requests
import base64
from io import BytesIO
url = 'https://ceesd.illinois.edu/wp-content/uploads/2021/02/ceesd_wordmark.svg'
#img = Image.open(BytesIO(response.content))

# drawing = svg2rlg('input/ice.svg')
# renderPM.drawToFile(drawing, 'output/ice.png', fmt='PNG')

def write_text(data: str, path: str):
    with open(path, 'w') as file:
        file.write(data)


svg = requests.get(url).text

write_text(svg, './NO_0301_Oslo.svg')


from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

drawing = svg2rlg('./NO_0301_Oslo.svg')
renderPM.drawToFile(drawing, "file.png", fmt="PNG")


def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


app.layout =  html.Div([ 
    html.Div([
        html.Div(
                    [
                        html.Img(src=b64_image("file.png"))
                        #html.H1("The Center for Exascale-enabled Scramjet Design", className="text-center", id="heading")
                    ], className = "col-md-6"
                ),
        ],className="row"),

    html.Div(
        [ #dropdown and score
        
        
            html.Div([
                html.Div(
                    [
                        dcc.Dropdown(
                            options=[
                                {'label': 'memory_usage_python', 'value': 'memory_usage_python'},
                                {'label': 't_step', 'value': 't_step'},  
                            ],
                            id='score-dropdown'
                        )
                    ], className="col-md-6"),
                html.Div(
                    html.Table(id='datatable', className = "table col-md-12")
            ),
            ],className="col-md-6"),

            html.Div(
                [ #Line Chart
                    dcc.Graph(id='line-graphs',
                        figure= subplots()
                              ),
                ], className = "col-md-6"
            ),
        ], className="row"),

    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id='line-graph',
                              figure = plot_line()
                            #  figure=go.Figure(
                            #      data=[
                            #          go.Scatter(
                            #             x=df_bubble.userscore,
                            #             y=df_bubble.metascore,
                            #             mode='lines',
                            #             text=df_bubble.genre,
                                        
                            #             marker=dict(
                            #             color= random.sample(range(1,200),15),
                            #             size=df_bubble.year,
                            #             sizemode='area',
                            #             sizeref=2.*max(df_bubble.year)/(40.**2),
                            #           sizemin=4
                            #             )
                            #          )
                                    
                            #      ],
                            #      layout=go.Layout(title="Genre popularity")
                            #  )
                              
                              
                              )
                ], className = "col-md-6"
            ),
   html.Div(
                [
                    dcc.Graph(id='bar-chart',
                              style={'margin-top': '20'})
                ], className = "col-md-6"
            ),
        
        ], className="row"),

 ], className="container-fluid")


##############################################################
            #DATA CONTROL (CONTROLLER)
##############################################################
@app.callback(
    Output(component_id='datatable', component_property='children'),
    [Input(component_id='score-dropdown', component_property='value')]
)

def update_table(input_value):
    return generate_table(df_table.sort_values([input_value], ascending=[False]).reset_index())

@app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    [Input(component_id='line-graph', component_property='hoverData')]
)

def update_graph(hoverData):
   return bar(hoverData)



if __name__ == '__main__':
    app.run_server(debug=True)

# if args.type == 'avg' and args.table is not None:
#     plot_bar_avg()
# elif args.type == 'sum' and args.table is not None:
#     plot_bar_sum()
# elif args.table is not None:
#     box_plot()
# else:
#     plot_all()


