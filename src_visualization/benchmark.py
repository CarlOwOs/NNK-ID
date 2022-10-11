import pandas as pd
import numpy as np
from ast import literal_eval
from statistics import median, mode

for ds in ["M1","M2","M3","M4","M5","M6","M7","M9","M10a","M10b","M10c",\
    "M10d","M11","M12","M13"]:

    df = pd.read_csv(f"experiments/results/a_knn_pca/{ds}/pca.csv")
    df = df[df["points"] == 2500]

    df["nnk_pc"] = df["nnk_pc"].apply(literal_eval)
    df["mean"] = df["nnk_pc"].apply(lambda x: sum(x)/len(x))
    df["median"] = df["nnk_pc"].apply(lambda x: median(x))
    df["mode"] = df["nnk_pc"].apply(lambda x: mode(x))

    df = df.drop(["nnk_pc", "knn_pc"], axis=1)
    df["ds"] = [ds]

    print(df)