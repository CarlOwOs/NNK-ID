import pandas as pd 
import altair as alt 
import numpy as np
from ast import literal_eval

for ds in ["test"]:#"M1","M2","M3","M4","M5","M6","M7","M9","M10a","M10b","M10c","M10d","M11","M12","M13"]:

    df = pd.read_csv(f"results/{ds}/sigma.csv")
    df["merge"] = df["points"].apply(lambda x: 2500-x)
    
    sm = alt.Chart(df).mark_line().encode(
        x=alt.X("merge:O", axis=alt.Axis(ticks=False, title="Merging Steps")),
        y=alt.Y("sigma:Q"),
    ).properties(title="Sigma vs. Merging on M1", width=450, height=250)

    sm.save(f"visualization/report/{ds}.html")
