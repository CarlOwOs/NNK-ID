import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

li = []

#for ds in ["a_scatter/M1"]:#,"hv_test/HV10","hv_test/HV100","benchmark_sample/M1","benchmark_sample/M2","benchmark_sample/M3","benchmark_sample/M4","benchmark_sample/M5","benchmark_sample/M6","benchmark_sample/M7","benchmark_sample/M9","benchmark_sample/M10a","benchmark_sample/M10b","benchmark_sample/M10c","benchmark_sample/M10d","benchmark_sample/M11","benchmark_sample/M12","benchmark_sample/M13"]:
for l, ed, ds in [("input","3072","0"), ("pool1","16384","1664840744"), ("pool2","8192","1664851323"), ("pool3","4096","1664854486"), ("pool4","2048","1664855871"), ("pool5","512","1664856355"), ("output","10","1664856955")]:

    df = pd.read_csv(f"a_cifar10_all/{ds}_pca.csv")
    df = df[df["points"] == 1000]
    df["nnk_pc"] = df["nnk_pc"].apply(literal_eval)
    df = df.drop(["knn_pc", "points"], axis=1)
    df["layer"] = [l]*len(df)
    df["embedding"] = [ed]*len(df)
    df = df.explode("nnk_pc")
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

sort_order = ["input","pool1","pool2","pool3","pool4","pool5","output"]

chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(
    x=alt.X("layer:N", sort=sort_order, axis=alt.Axis(title="")),
    y=alt.Y("nnk_pc:Q", axis=alt.Axis(title="NNK Principal Components")),
).properties(title="NNK PCs by Layer", width=350)

chart.save(f"visualization/_layers/layers.html")

