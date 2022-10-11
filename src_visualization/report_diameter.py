import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

for target, ds in [("plane","01"),("car","02"),("bird","03"),("cat","04"),("deer","05"),("dog","06"),("frog","07"),("horse","08"),("ship","09"),("truck","10")]:

    df = pd.read_csv(f"_classes/{ds}/diameter.csv")
    df = df[df["points"] == 1000]

    df["diameter"] = df["diameter"].apply(literal_eval)\
        .apply(lambda x: [item for item in x])

    # histogram bin range - maybe find top percentile or smth
    x = df["diameter"].tolist()
    percentile = np.percentile([item for sublist in x for item in sublist], 100)
    top_range = percentile

    df["counts"] = df["diameter"].apply(lambda x: np.histogram(x, bins=10, range=(0,2))[0]).apply(lambda x: x/sum(x))
    df["bins"] = df["diameter"].apply(lambda x: np.histogram(x, bins=10, range=(0,2))[1])
    df["bins"] = df["bins"].apply(lambda x: [f"{round(x[i],2)}-{round(x[i+1],2)}" for i in range(len(x)-1)])

    df = df.explode(["counts", "bins"])

    sm = alt.Chart(df).mark_bar().encode(
        x=alt.X("bins:O", axis=alt.Axis(title="Polytope Diameter")),
        y=alt.Y("counts:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="")),
        tooltip=["counts:Q"],
    ).properties(title=f"{target}", width=200, height=150) # N = 2500 N = 500

    sm.save(f"visualization/report/{ds}.html")