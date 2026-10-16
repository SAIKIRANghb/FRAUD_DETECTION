"""Build Elliptic node table with graph features and labels."""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, set_seed

cfg = load_config()
set_seed(cfg["seed"])

def main():
    t0 = time.time()
    features = pd.read_parquet("data/elliptic_features.parquet")
    classes = pd.read_parquet("data/elliptic_classes.parquet")
    edges = pd.read_parquet("data/elliptic_edges.parquet")

    # features: col0 txId, col1 time_step, rest features
    tx_ids = features.iloc[:, 0].astype(int).values
    time_step = features.iloc[:, 1].astype(int).values
    Xraw = features.iloc[:, 2:].astype(float)
    # column names
    feat_cols = [f"f_{i}" for i in range(Xraw.shape[1])]
    df = pd.DataFrame({"txId": tx_ids, "time_step": time_step})
    for i, c in enumerate(feat_cols):
        df[c] = Xraw.iloc[:, i].values

    # merge labels
    classes = classes.rename(columns={"class": "cls"})
    classes["txId"] = classes["txId"].astype(int)
    classes["cls"] = classes["cls"].astype(str)
    df = df.merge(classes, on="txId", how="left")
    df["is_fraud"] = (df["cls"] == "1").astype(int)   # illicit
    df["is_licit"] = (df["cls"] == "2").astype(int)
    df["labeled"] = df["cls"].isin(["1", "2"]).astype(int)

    # Build graph
    G = nx.DiGraph()
    # add all nodes
    G.add_nodes_from(tx_ids.tolist())
    for _, r in edges.iterrows():
        u = int(r.txId1); v = int(r.txId2)
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    print("Graph:", G.number_of_nodes(), G.number_of_edges(), flush=True)
    Gu = G.to_undirected()

    # degrees
    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())
    out_deg_w = dict(G.out_degree(weight="weight"))
    in_deg_w = dict(G.in_degree(weight="weight"))

    # pagerank (directed)
    try:
        pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=50, tol=1e-6)
        print("pagerank done", flush=True)
    except Exception as e:
        print("pagerank fail", e, flush=True)
        pr = {n: 0.0 for n in G}

    # connected components on undirected
    comp_size = {}
    for c in nx.connected_components(Gu):
        sz = len(c)
        for node in c:
            comp_size[node] = sz
    for n in G:
        comp_size.setdefault(n, 1)

    # clustering coeff (undirected)
    try:
        clustering = nx.clustering(Gu)
        print("clustering done", flush=True)
    except Exception as e:
        print("clustering fail", e, flush=True)
        clustering = {n: 0.0 for n in G}

    # community (Louvain) with fallback
    comm_id = {}
    comm_size = {}
    try:
        comms = nx.community.louvain_communities(Gu, seed=42, resolution=1.0)
        for idx, c in enumerate(comms):
            for node in c:
                comm_id[node] = idx
                comm_size[node] = len(c)
        print("louvain done:", len(comms), flush=True)
    except Exception as e:
        print("louvain fail", e, flush=True)
        for n in G:
            comm_id[n] = 0
            comm_size[n] = len(G)

    # unique partners
    out_partners = defaultdict(set)
    in_partners = defaultdict(set)
    for u, v in G.edges():
        out_partners[u].add(v)
        in_partners[v].add(u)

    # reciprocity pairs / cycle
    mutual_pairs = set()
    for u, v in list(G.edges()):
        if G.has_edge(v, u):
            mutual_pairs.add(tuple(sorted((u, v))))
    mutual_nodes = set()
    for u, v in mutual_pairs:
        mutual_nodes.add(u); mutual_nodes.add(v)

    # build graph feature frame aligned to df order
    id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}
    rows_graph = []
    for tid in tx_ids:
        up = len(out_partners.get(tid, set()))
        ip = len(in_partners.get(tid, set()))
        total_c = (out_deg.get(tid, 0) + in_deg.get(tid, 0))
        rows_graph.append({
            "txId": tid,
            "g_out_degree": out_deg.get(tid, 0),
            "g_in_degree": in_deg.get(tid, 0),
            "g_total_degree": total_c,
            "g_weighted_out": out_deg_w.get(tid, 0),
            "g_weighted_in": in_deg_w.get(tid, 0),
            "g_pagerank": pr.get(tid, 0.0),
            "g_clustering": clustering.get(tid, 0.0),
            "g_component_size": comp_size.get(tid, 1),
            "g_community_id": comm_id.get(tid, -1),
            "g_community_size": comm_size.get(tid, 1),
            "g_unique_out": up,
            "g_unique_in": ip,
            "g_partner_diversity": (up + ip) / max(total_c, 1),
            "g_mutual_flag": int(tid in mutual_nodes),
        })
    gdf = pd.DataFrame(rows_graph)
    df = df.merge(gdf, on="txId", how="left").fillna(0)

    # merge graph features done; remaining columns stay as-is
    # Save
    out = cfg["data"]["output_path"]  # currently accounts path; override
    out = "data/elliptic_processed.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out.replace(".parquet", ".csv"), index=False)
    with open("data/elliptic_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("Processed:", df.shape, "labeled:", df["labeled"].sum(), "fraud:", df["is_fraud"].sum(), flush=True)
    print("Time:", round(time.time()-t0, 1), "s", flush=True)

if __name__ == "__main__":
    main()
