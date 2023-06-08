import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted, venn2_circles, venn3, venn3_circles
from collections import OrderedDict
import numpy as np
import math
import json

st.set_page_config(layout="wide")

table_list = [
    # "aspirin_0.4_selfies.xlsx",
    # "sas_3_selfies.xlsx",
    "aspirin_0.4.xlsx",
    # "sas_3.xlsx",
]


def highlight_row(x, selected_row):
    if x.name == selected_row:
        return ['background-color: #ddeed5'] * len(x)
    else:
        return [''] * len(x)
    

def roundup(x):
    return int(round(x/1_000_000))   
 
    
def read_file(filename: str) -> list:
    with open(filename, "r", encoding="utf-8") as f:
        data_arr = f.read().splitlines()
    
    return data_arr
    

def run():
    st.header("Exploring Molecular Generation with the GDB-13 dataset") \

    with st.sidebar:
        st.write("Experiment Config")

        selected_table = st.selectbox(
            "Select table (1)",
            table_list
        )

    # Table
    df = pd.read_excel(f"old_Sampling_results_{selected_table}").set_index("Model")

    with st.sidebar:
        selected_row = st.selectbox(
            "Select row (1)",
            df.index.values.tolist()
        )

        compare = st.checkbox('Compare 2 tables', True) 

    if compare:
        with st.sidebar:
            st.write("Table 2")
            selected_table_2 = st.selectbox(
                "Select table (2)",
                table_list
            )

        # Table
        df_2 = pd.read_excel(f"old_Sampling_results_{selected_table_2}").set_index("Model")

        with st.sidebar:
            selected_row_2 = st.selectbox(
                "Select row (2)",
                df_2.index.values.tolist()
            )       

    # Table
    # df = df.style.apply(lambda x: ['background: red'] if x==selected_row, axis=1)
    # df = df.style.apply(highlight_row, axis=1)

    # def _apply_prop(_, style, selected_row):
    #     return style.applymap(lambda x: ['background: red'] if x==selected_row else "")

    # df.style.apply(_apply_prop, axis=0, style=df, selected_row=selected_row)
    
    # # df.style.highlight_between([row for row in df.index if row == selected_row], axis=0)
    # st.dataframe(df.style.apply(_apply_prop, axis=0, style=df, prop=selected_row))
    
    st.dataframe(df)

    if compare:
        st.dataframe(df_2)
        row_2 = df_2.loc[selected_row_2]

    row = df.loc[selected_row]

    with open("Generated_prop_stats.json") as json_file:
        prop_dict = json.load(json_file)


    def show_fig(selected_table, row, header, header_sub, mol_type, fig, ax):
        v = venn3(subsets=({"100":30, "010":200, "110": 50, "101": 20, "001": 1, "011": 20, "111": 30}), set_labels = ('A', 'B', 'C'), ax=ax)
        c = venn3_circles(subsets=({"100":30, "010": 200, "110": 50, "101": 20, "001": 1, "011": 20, "111": 30}), linestyle='dashed', ax=ax)

        # Line style    
        c[0].set_ls('dashed')
        c[1].set_ls('dotted')
        c[2].set_ls('solid')


        # Unknown subset 
        v.get_patch_by_id('100').set_alpha(1.0)
        v.get_patch_by_id('100').set_color('white')
        

        # Counts
        v.get_label_by_id('100').set_text('Unknown')
        v.get_label_by_id('010').set_text("1B")
        v.get_label_by_id('001').set_text(row["Valid Smiles Unique"] - row["In_GDB13 & Subset Unique"] - (row["In_Subset Unique"] + row["In_GDB13 Unique"] - 2* row["In_GDB13 & Subset Unique"] ))
        v.get_label_by_id('111').set_text(row["In_GDB13 & Subset Unique"])
        v.get_label_by_id('110').set_text(str(roundup(row["Subset count"])) + "M - " + str(row["In_GDB13 & Subset Unique"]))
        v.get_label_by_id('101').set_text(row["In_Subset Unique"] - row["In_GDB13 & Subset Unique"])
        v.get_label_by_id('011').set_text(row["In_GDB13 Unique"] - row["In_GDB13 & Subset Unique"])


        # Labels
        v.get_label_by_id('A').set_text(header_sub + " set")
        v.get_label_by_id('B').set_text("GDB-13")
        v.get_label_by_id('C').set_text("Generated")


        st.subheader(f"Generated score distribution for {header_sub} :blue[{mol_type}]")
        name = selected_table.split(".xlsx")[0]
        prop_tabel_dict = prop_dict["./Generations_" + name + "/" + row.name + ".csv"]["repetitions"]["all_scores"]
        prop_tabel_unique_dict = prop_dict["./Generations_" + name + "/" + row.name + ".csv"]["unique"]["all_scores"]
        prop_df = pd.DataFrame({"scores": prop_tabel_dict.keys(), "freqs": prop_tabel_dict.values()})
        prop_unique_df = pd.DataFrame({"scores": prop_tabel_unique_dict.keys(), "freqs": prop_tabel_unique_dict.values()})
        st.bar_chart(prop_df, x="scores", y="freqs")
        
        
        st.subheader(f"Generated score distribution for {header_sub} (Unique) :blue[{mol_type}]")
        st.bar_chart(prop_unique_df, x="scores", y="freqs")

        # st.subheader(f"Cumulative sum for unique values")
        # smiles_arr = read_file("./Generations_" + name + "/" + row.name + ".csv")
        
        # unique_cum = {}
        # for i in range(1, 1000000, 1000):
        #     unique_cum[i] = len(set(smiles_arr[i:i+1000]))

        # st.dataframe(pd.DataFrame(unique_cum))



        st.subheader(header)

        return fig

        

    col1, col2 = st.columns(2)    

    mol_type = "Selfies" if "selfies" in selected_table else "Smiles"
    header_sub = "Aspirin >= 0.4" if "aspirin_0.4" in selected_table else "Sas <= 3" 

    if compare:
        mol_type_2 = "Selfies" if "selfies" in selected_table_2 else "Smiles"
        header_sub_2 = "Aspirin >= 0.4" if "aspirin_0.4" in selected_table_2 else "Sas <= 3"  

        fig, ax = plt.subplots(1,2, figsize=(10,15))
        with col1:
            header = f"{mol_type} Unique counts for {header_sub}"
            fig = show_fig(selected_table, row, header, header_sub, mol_type, fig, ax[0]) 
        
        with col2:
            header_2 = f"{mol_type_2} Unique counts for {header_sub_2}"
            fig = show_fig(selected_table_2, row_2, header_2, header_sub_2, mol_type_2, fig, ax[1])


        st.pyplot(fig)    
    else:
        fig, ax = plt.subplots(1, figsize=(10,15))
        fig = show_fig(selected_table, row, header, header_sub, fig, ax) 
        st.pyplot(fig) 



    # with open("Generated_prop_stats.json") as json_file:
    #     prop_dict = json.load(json_file)
    #     name = selected_table.split(".xlsx")[0]
    #     prop_tabel_dict = prop_dict["./Generations_" + name + "/" + selected_row + ".csv"]


    # prop_df_1 = pd.DataFrame(prop_tabel_dict)
    # st.dataframe(prop_df_1["unique"]["all_scores"])
    # fig, ax = plt.subplots()

    # plt.bar(range(len(prop_dict.keys())), list(prop_dict.values()), align='center')
    # plt.xticks(range(len(prop_dict)), list(prop_dict.keys()))    



    fig, ax = plt.subplots(1,2, figsize=(10,15))
    matplotlib.rcParams.update({'font.size': 8})
    # First square
    ax[0].plot([1, 3, 3, 1, 1], [2, 2, 4, 4, 2], color="black")
    ax[0].plot([1,3], [2.75, 2.75], color="black")
    ax[0].plot([1,3], [2.2, 2.2], color="black")

    ax[0].fill_between([1,3], 2, 4,  color='#ffbb78', alpha=0.5)
    ax[0].fill_between([2,4], 1.5, 3.5,  color='#aec7e8', alpha=0.5)
    ax[0].fill_between([2,3], 2, 3.5,  color='#99cc99', alpha=0.5)


    ax[0].text(1.6, 3.5, '1B', ha='center', va='center')
    ax[0].text(1.25, 3.93, "GDB-13", ha='center', va='center', fontsize="x-small", color="black")
    ax[0].text(2.3, 3.1, row["In_GDB13"] - row["In_GDB13 & Subset"], ha='center', va='center')
    ax[0].text(2.8, 3.1, row["In_GDB13 Unique"] - row["In_GDB13 & Subset Unique"], ha='center', va='center', color="red")
    
    
    # Subset
    ax[0].text(1.6, 2.4, str(roundup(row["Subset count"])) + "M", ha='center', va='center')
    ax[0].text(1.38, 2.65, header_sub, ha='center', va='center', fontsize="x-small", color="black")
    ax[0].text(2.3, 2.4, row["In_GDB13 & Subset"] - row["Train Recall"], ha='center', va='center')
    ax[0].text(2.8, 2.4, row["In_GDB13 & Subset Unique"]-row["Train Recall Unique"], ha='center', va='center', color="red")
    

    # Tain
    ax[0].text(1.6, 2.1, '1M', ha='center', va='center')
    ax[0].text(1.25, 2.1, "Train set", ha='center', va='center', fontsize="x-small", color="black")
    ax[0].text(2.3, 2.1, row["Train Recall"], ha='center', va='center')
    ax[0].text(2.8, 2.1, row["Train Recall Unique"], ha='center', va='center', color="red")


    # Second square
    ax[0].plot([2, 4, 4, 2, 2], [1.5, 1.5, 3.5, 3.5, 1.5], color="black")
    ax[0].text(3.7, 3.4, "Generated", ha='center', va='center', fontsize="x-small", color="black")
    ax[0].text(3.6, 2.65, header_sub, ha='center', va='center', fontsize="x-small", color="black")
    ax[0].text(3.75, 1.6, "Invalid", ha='center', va='center', fontsize="x-small", color="black")
    ax[0].plot([3, 4], [2, 2], color="black")
    ax[0].plot([3, 4], [2.75, 2.75], color="black")
    ax[0].text(3.4, 2.3, row["In_Subset"] - row["In_GDB13 & Subset"], ha='center', va='center')
    ax[0].text(3.8, 2.3, row["In_Subset Unique"] - row["In_GDB13 & Subset Unique"], ha='center', va='center', color="red")


    # Invalid Smiles count
    ax[0].text(3, 1.7, row["Generated Smiles count"] - row["Valid Smiles"], ha='center', va='center')


    # Valid Smiles count
    ax[0].text(3.4, 3.1, row["Valid Smiles"] - row["In_GDB13 & Subset"] - (row["In_Subset"] + row["In_GDB13"] - 2*row["In_GDB13 & Subset"]), ha='center', va='center')
    ax[0].text(3.8, 3.1, row["Valid Smiles Unique"] - row["In_GDB13 & Subset Unique"] - (row["In_Subset Unique"] + row["In_GDB13 Unique"] - 2*row["In_GDB13 & Subset Unique"]), ha='center', va='center', color="red")
        

    ax[0].set_title(f"{mol_type} counts for {header_sub}")

    #######################################################################################################################

    # First square
    ax[1].plot([1, 3, 3, 1, 1], [2, 2, 4, 4, 2], color="black")
    ax[1].plot([1,3], [2.75, 2.75], color="black")
    ax[1].plot([1,3], [2.2, 2.2], color="black")

    ax[1].fill_between([1,3], 2, 4,  color='#ffbb78', alpha=0.5)
    ax[1].fill_between([2,4], 1.5, 3.5,  color='#aec7e8', alpha=0.5)
    ax[1].fill_between([2,3], 2, 3.5,  color='#99cc99', alpha=0.5)


    ax[1].text(1.6, 3.5, '1B', ha='center', va='center')
    ax[1].text(1.25, 3.93, "GDB-13", ha='center', va='center', fontsize="x-small", color="black")
    ax[1].text(2.3, 3.1, row_2["In_GDB13"] - row_2["In_GDB13 & Subset"], ha='center', va='center')
    ax[1].text(2.8, 3.1, row_2["In_GDB13 Unique"] - row_2["In_GDB13 & Subset Unique"], ha='center', va='center', color="red")
    
    
    # Subset
    ax[1].text(1.6, 2.4, str(roundup(row_2["Subset count"])) + "M", ha='center', va='center')
    ax[1].text(1.38, 2.65, header_sub_2, ha='center', va='center', fontsize="x-small", color="black")
    ax[1].text(2.3, 2.4, row_2["In_GDB13 & Subset"] - row_2["Train Recall"], ha='center', va='center')
    ax[1].text(2.8, 2.4, row_2["In_GDB13 & Subset Unique"]-row_2["Train Recall Unique"], ha='center', va='center', color="red")
    

    # Tain
    ax[1].text(1.6, 2.1, '1M', ha='center', va='center')
    ax[1].text(1.25, 2.1, "Train set", ha='center', va='center', fontsize="x-small", color="black")
    ax[1].text(2.3, 2.1, row_2["Train Recall"], ha='center', va='center')
    ax[1].text(2.8, 2.1, row_2["Train Recall Unique"], ha='center', va='center', color="red")


    # Second square
    ax[1].plot([2, 4, 4, 2, 2], [1.5, 1.5, 3.5, 3.5, 1.5], color="black")
    ax[1].text(3.7, 3.4, "Generated", ha='center', va='center', fontsize="x-small", color="black")
    ax[1].text(3.6, 2.65, header_sub_2, ha='center', va='center', fontsize="x-small", color="black")
    ax[1].text(3.75, 1.6, "Invalid", ha='center', va='center', fontsize="x-small", color="black")
    ax[1].plot([3, 4], [2, 2], color="black")
    ax[1].plot([3, 4], [2.75, 2.75], color="black")
    ax[1].text(3.4, 2.3, row_2["In_Subset"] - row_2["In_GDB13 & Subset"], ha='center', va='center')
    ax[1].text(3.8, 2.3, row_2["In_Subset Unique"] - row_2["In_GDB13 & Subset Unique"], ha='center', va='center', color="red")


    # Invalid Smiles count
    ax[1].text(3, 1.7, row_2["Generated Smiles count"] - row_2["Valid Smiles"], ha='center', va='center')


    # Valid Smiles count
    ax[1].text(3.4, 3.1, row_2["Valid Smiles"] - row_2["In_GDB13 & Subset"] - (row_2["In_Subset"] + row_2["In_GDB13"] - 2*row_2["In_GDB13 & Subset"]), ha='center', va='center')
    ax[1].text(3.8, 3.1, row_2["Valid Smiles Unique"] - row_2["In_GDB13 & Subset Unique"] - (row_2["In_Subset Unique"] + row_2["In_GDB13 Unique"] - 2*row_2["In_GDB13 & Subset Unique"]), ha='center', va='center', color="red")
        



    ax[1].set_title(f"{mol_type_2} counts for {header_sub_2}")

    


    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].axis('off')
    ax[1].axis('off')
    st.pyplot(fig)


run()