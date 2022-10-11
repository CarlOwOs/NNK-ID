import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

#for ds, id in [("M1",10),("M2",3),("M3",4),("M4",4),("M5",2),("M6",6),("M7",2),("M9",20),("M10a",10),("M10b",17),("M10c",24),("M10d",70),("M11",2),("M12",20),("M13",1)]:
#for l, ed, ds in [("input","3072","0"), ("pool1","16384","1664840744"), ("pool2","8192","1664851323"), ("pool3","4096","1664854486"), ("pool4","2048","1664855871"), ("pool5","512","1664856355"), ("output","10","1664856955")]:
for target, ds in [("plane","01"),("car","02"),("bird","03"),("cat","04"),("deer","05"),("dog","06"),("frog","07"),("horse","08"),("ship","09"),("truck","10")]:

    df = pd.read_csv(f"_classes/{ds}/neighbors.csv")
    df["neighbors"] = df["neighbors"].apply(literal_eval)
    df = df.explode("neighbors")

    line = alt.Chart(df).mark_line().encode(
        x=alt.X('points', sort = 'descending'),
        y='mean(neighbors)'
    )

    band = alt.Chart(df).mark_errorband(extent='stdev').encode(
        x = alt.X('points', sort = 'descending'),
        y = alt.Y('neighbors', axis=alt.Axis(title='NNK Neighbors')),
    )
    chart = (band+line).properties(title=f"{target}")
    chart.save(f"visualization/__cifar_classes/{target}_neighbors.html")

    #####

    df = pd.read_csv(f"_classes/{ds}/pca.csv")
    df["KNN"] = df["knn_pc"].apply(literal_eval)
    df["NNK"] = df["nnk_pc"].apply(literal_eval)
    df = df.drop(["knn_pc", "nnk_pc"], axis=1)
    df = df.melt("points")
    df = df.explode("value")
    df["Neighborhood"] = df["variable"]
    
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('points', sort = 'descending'),
        y='mean(value)',
        color='Neighborhood'
    )
    
    # line2 = alt.Chart(pd.DataFrame({'y': [id]})).mark_rule().encode(y='y')

    band = alt.Chart(df).mark_errorband(extent='stdev').encode(
        x = alt.X('points', sort = 'descending'),
        y = alt.Y('value', axis=alt.Axis(title='Principal Components')),
        color = 'Neighborhood'
    )

    chart = (band+line).properties(title=f"{target}")

    chart.save(f"visualization/__cifar_classes/{target}_pca.html")
