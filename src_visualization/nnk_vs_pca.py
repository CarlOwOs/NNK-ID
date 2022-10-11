import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

charts = []
charts_2500 = []
#for ds in ["isomap","mnist"]:#["M1","M2","M3","M4","M5","M6","M7","M9","M10a","M10b","M10c","M10d","M11","M12","M13"]:
for l, ds in enumerate(["1664840744", "1664851323", "1664854486", "1664855871", "1664856355", "1664856832", "1664856955"]):
    
    df = pd.read_csv(f"a_cifar10_all/{ds}_scatter.csv")
    df["neighbors"] = df["neighbors"].apply(literal_eval).apply(lambda x: sum(x)/len(x))
    df["nnk_pc"] = df["nnk_pc"].apply(literal_eval).apply(lambda x: sum(x)/len(x))
    df["dataset"] = [l]*len(df)

    df = df[df["points"] > 100]

    scatter = alt.Chart(df).mark_circle().encode(
        x=alt.X("nnk_pc:Q"),
        y=alt.Y("neighbors:Q"),
        color="dataset:N",
        size="points:Q",
        tooltip=["dataset:N", "neighbors:Q", "nnk_pc:Q", "points:Q"]
    )
    scatter_2500 = alt.Chart(df[df["points"] == 1000]).mark_circle().encode(
        x=alt.X("nnk_pc:Q"),
        y=alt.Y("neighbors:Q"),
        color="dataset:N",
        size="points:Q",
        tooltip=["dataset:N", "neighbors:Q", "nnk_pc:Q", "points:Q"]
    )

    charts += [scatter]
    charts_2500 += [scatter_2500]

## <- tab for aggregate
    # 2^dim
x = np.arange(max(0, min(df["nnk_pc"])-5),max(df["nnk_pc"])+1,0.1)
y = [2**v for v in x]
df = pd.DataFrame({"x":x,"y":y})
line = alt.Chart(df).mark_line(clip=True).encode(
    x = alt.X("x:Q"),
    y = alt.Y("y:Q", scale=alt.Scale(domain=(0, 100)))
)

chart = line + scatter
for ch in charts:
    chart += ch

chart_2500 = line 
for ch in charts_2500:
    chart_2500 += ch

chart_f = chart_2500.properties(title="NNK Neighbors vs. PCs without Merging") | chart.properties(title="NNK Neighbors vs. PCs Merging")
chart = chart.properties(title="NNK Neighbors vs. PCs Merging")
chart.save(f"visualization/_cifar10/all_nnk_vs_pca.html")