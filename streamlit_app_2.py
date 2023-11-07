import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted, venn2_circles, venn3, venn3_circles
from collections import OrderedDict
import numpy as np
import math
import json

# st.set_page_config(layout="wide")

TABLE_LIST = [
    "aspirin_0.4_sf_302M",
    "aspirin_0.4_sf_85M",
    "aspirin_0.4_sf_800K",
]

COLOR_LIST = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                 '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                 '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                 '#17becf', '#9edae5']


def extract_value(file_name, start_str, end_str):
    start_idx = file_name.index(start_str) + len(start_str)
    end_idx =  file_name.index(end_str)
    
    value = file_name[start_idx: end_idx]

    # remove 0-s at the end
    value = f'{float(value):g}'
    
    return value
    

def run():
    st.header("Exploring Molecular Generation with the GDB-13 dataset")

    with st.sidebar:
        st.write("Experiment Config")

        selected_model_1 = st.selectbox(
            "Select model",
            TABLE_LIST
        )

    # Table
    df_1 = pd.read_excel(f"./ablations/statistics/Sampling_results_{selected_model_1}.xlsx")
    df_1['epoch'] = df_1['Model'].apply(lambda x: extract_value(x, "ep_", "_temp"))
    df_1['temp'] = df_1['Model'].apply(lambda x: extract_value(x, "temp_", "_gen"))
    df_1['color'] = "black"

    df_1 = df_1.sort_values("temp")

    # values
    gen_list_1 = df_1["Generated Smiles count"].unique()

    with st.sidebar:
        selected_gen_count_1 = st.selectbox(
            "Select generation count",
            gen_list_1
        )

        # df_1 = df_1[df_1["Generated Smiles count"] == selected_gen_count_1]
        df_1 = df_1[df_1["Generated Smiles count"] == 10000]
        epoch_list_1 = df_1["epoch"].unique()
    
    df_by_epoch = {}

    for i, epoch in enumerate(epoch_list_1):
        df_by_epoch[epoch] =  df_1[df_1["epoch"] == epoch]
        df_by_epoch[epoch]["color"] =  COLOR_LIST[i]

    st.dataframe(df_1) 

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.suptitle(f"Unique Counts at Different Temperatures [{selected_model_1}]", fontsize=18)

    tables = [
        [(0,0), "Canonical Smiles Unique"], 
        [(0,1), "In_Subset Unique"], 
        [(1,0), "In_GDB13 Unique"], 
        [(1,1), "In_GDB13 & Subset Unique"], 
    ]


    for i, title in tables:
        axs[i].set_title(title)

        for epoch in epoch_list_1:
            axs[i].plot(df_by_epoch[epoch]["temp"], df_by_epoch[epoch][title], label=f"Epoch {epoch}", marker="o", markersize=3, c=df_by_epoch[epoch]["color"].iloc[0], linestyle="dotted")
            # axs[i].scatter(df_by_epoch[epoch]["temp"], df_by_epoch[epoch][title], label=f"Epoch {epoch}", c=df_by_epoch[epoch]["color"].iloc[0])

        # axs[i].set_ylim([7000, 10050])
        # axs[i].set_xticks(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
        axs[i].set_xticks(['0.4', '0.625','0.8', '1'])

    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 2.2));  
    st.pyplot(fig)

run()    