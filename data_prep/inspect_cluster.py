import pandas as pd

pd.set_option("display.max_colwidth", None)

df = pd.read_csv("usv_clusters/cluster_assignments.csv")

cluster = 7

cluster_df = df[df["cluster"] == cluster]

print(cluster_df[["source_audio", "call_index", "start_time_s", "end_time_s"]])