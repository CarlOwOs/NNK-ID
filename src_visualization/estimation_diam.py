import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

#for ds in ["nnk_two"]:
#for l, ds in enumerate(["1664840744", "1664851323", "1664854486", "1664855871", "1664856355", "1664856832", "1664856955"]):
for target, ds in [("plane","01"),("car","02"),("bird","03"),("cat","04"),("deer","05"),("dog","06"),("frog","07"),("horse","08"),("ship","09"),("truck","10")]:

    df = pd.read_csv(f"_classes/{ds}/diameter.csv")
    
    df = df[df["points"]%(len(df)//9) == 0]

    df["diameter"] = df["diameter"].apply(literal_eval)\
        .apply(lambda x: [item for item in x])

    # histogram bin range - maybe find top percentile or smth
    x = df["diameter"].tolist()
    percentile = np.percentile([item for sublist in x for item in sublist], 99)
    top_range = percentile

    df["counts"] = df["diameter"].apply(lambda x: np.histogram(x, bins=10, range=(0,top_range))[0]).apply(lambda x: x/sum(x))
    df["bins"] = df["diameter"].apply(lambda x: np.histogram(x, bins=10, range=(0,top_range))[1])
    df["bins"] = df["bins"].apply(lambda x: [f"{round(x[i],2)}-{round(x[i+1],2)}" for i in range(len(x)-1)])

    df = df.explode(["counts", "bins"])

    sm = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O", axis=alt.Axis(title="Values")),
        y=alt.Y("counts:Q", axis=alt.Axis(title="")),
        facet=alt.Facet('points:O', columns=3, sort="descending")
    ).properties(title=f"{target}", width=150, height=150)

    sm.save(f"visualization/__cifar_classes/{target}_diam.html")