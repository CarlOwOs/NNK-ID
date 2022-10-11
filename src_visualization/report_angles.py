import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

for target, ds in [("plane","01"),("car","02"),("bird","03"),("cat","04"),("deer","05"),("dog","06"),("frog","07"),("horse","08"),("ship","09"),("truck","10")]:

    df = pd.read_csv(f"_classes/{ds}/angles.csv")
    df = df[df["points"] == 1000]
    
    df["nnk_angles"] = df["nnk_angles"].apply(literal_eval)\
        .apply(lambda x: [item for sublist in x for item in sublist])\
        .apply(lambda x: np.histogram(x, bins=9, range=(0,90))[0]).apply(lambda x: x/sum(x))

    df["rnd_angles"] = df["rnd_angles"].apply(literal_eval)\
        .apply(lambda x: [item for sublist in x for item in sublist])\
        .apply(lambda x: np.histogram(x, bins=9, range=(0,90))[0]).apply(lambda x: x/sum(x))

    df["bins"] = [["0-10","10-20","20-30","30-40","40-50","50-60","60-70"\
        ,"70-80","80-90"]]*len(df)
    
    df = df.explode(["nnk_angles", "rnd_angles", "bins"])

    sm = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O", axis=alt.Axis(ticks=False, labels=False, title="Adjacent NNK Neighborhoods")),
        y=alt.Y("nnk_angles:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["nnk_angles:Q"],
    ).properties(title=f"{target}", width=200, height=150)

    sm2 = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O", axis=alt.Axis(ticks=False, labels=False, title="Random NNK Neighborhoods")),
        y=alt.Y("rnd_angles:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["rnd_angles:Q"],
    ).properties(title=f"{target}", width=200, height=150)

    sm.save(f"visualization/__cifar_classes/{target}_nnk.html")
    sm2.save(f"visualization/__cifar_classes/{target}_rnd.html")