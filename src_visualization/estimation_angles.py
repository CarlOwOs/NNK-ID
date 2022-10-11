import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

#for ds in ["isomap"]:#"M1","M2","M3","M4","M5","M6","M7","M9","M10a","M10b","M10c","M10d","M11","M12","M13"]:
#for l, ds in enumerate(["1664840744", "1664851323", "1664854486", "1664855871", "1664856355", "1664856832", "1664856955"]):
for target, ds in [("plane","01"),("car","02"),("bird","03"),("cat","04"),("deer","05"),("dog","06"),("frog","07"),("horse","08"),("ship","09"),("truck","10")]:

    df = pd.read_csv(f"_classes/{ds}/angles.csv")
    
    df = df[df["points"]%(len(df)//9) == 0]

    df["nnk_angles"] = df["nnk_angles"].apply(literal_eval)\
        .apply(lambda x: [item for sublist in x for item in sublist])\
        .apply(lambda x: np.histogram(x, bins=9, range=(0,90))[0])

    df["rnd_angles"] = df["rnd_angles"].apply(literal_eval)\
        .apply(lambda x: [item for sublist in x for item in sublist])\
        .apply(lambda x: np.histogram(x, bins=9, range=(0,90))[0])

    df["bins"] = [["0-10","10-20","20-30","30-40","40-50","50-60","60-70"\
        ,"70-80","80-90"]]*len(df)
    
    df = df.explode(["nnk_angles", "rnd_angles", "bins"])

    sm = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O"),
        y=alt.Y("nnk_angles:Q"),
        tooltip=["nnk_angles:Q"],
        facet=alt.Facet('points:O', columns=3, sort="descending")
    ).properties(title=f"{target}", width=150, height=150)

    sm2 = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O"),
        y=alt.Y("rnd_angles:Q"),
        tooltip=["rnd_angles:Q"],
        facet=alt.Facet('points:O', columns=3, sort="descending")
    ).properties(title=f"{target}", width=150, height=150)

    sm.save(f"visualization/__cifar_classes/{target}_nnk.html")
    sm2.save(f"visualization/__cifar_classes/{target}_rnd.html")