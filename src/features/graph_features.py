"""Compute graph-theoretic features per account (vertex) from transaction graph.

Mimics PayPal's graph analysis: connectivity, communities, centrality, etc.
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config


def build_graph(txs):
    G = nx.DiGraph()
    # adjacency aggregation
    edge_weights = defaultdict(lambda: [0, 0.0])  # count, amount
    for _, r in txs.iterrows():
        key = (int(r.sender_id), int(r.receiver_id))
        edge_weights[key][0] += 1
        edge_weights[key][1] += float(r.amount)

    for (u, v), (c, a) in edge_weights.items():
        G.add_edge(u, v, weight=c, amount=a)
    # ensure all accounts present
    accounts = set(txs.sender_id.unique()) | set(txs.receiver_id.unique())
    for a in accounts:
        if a not in G:
            G.add_node(a)
    return G


def compute_graph_features(accounts_df, txs):
    cfg = load_config()
    G = build_graph(txs)
    n = accounts_df["account_id"].max() + 1
    # undirected for community/centrality where sensible
    Gu = G.to_undirected()

    # basic degrees
    out_deg = dict(G.out_degree(weight="weight"))
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg_u = dict(G.out_degree())
    in_deg_u = dict(G.in_degree())
    total_deg = {k: out_deg_u.get(k, 0) + in_deg_u.get(k, 0) for k in range(n)}

    # pagerank (directed)
    try:
        pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=100)
    except Exception:
        pr = {i: 0.0 for i in range(n)}

    # clustering coefficient on undirected
    clustering = nx.clustering(Gu)

    # connected components sizes
    comp = {}
    comp_size = {}
    for c in nx.connected_components(Gu):
        sz = len(c)
        for node in c:
            comp[node] = sz
    for i in range(n):
        comp_size[i] = comp.get(i, 1)

    # community detection (Louvain)
    try:
        communities = nx.community.louvain_communities(Gu, seed=42)
    except Exception:
        communities = [set(range(n))]
    comm_id = {}
    comm_size = {}
    for idx, c in enumerate(communities):
        for node in c:
            comm_id[node] = idx
            comm_size[node] = len(c)

    # reciprocity / back-and-forth partners
    recip_partners = defaultdict(int)
    pair_counts = Counter()
    for (u, v), (c, a) in defaultdict(int).__class__ and {}.__class__ and {} if False else {}.items():
        pass
    # compute mutual edges
    mutual = 0
    ababa_accounts = set()
    # detect ABABA pattern: pairs with >=2 round trips
    pairs = set()
    for u, v in G.edges():
        if G.has_edge(v, u):
            pair = tuple(sorted((u, v)))
            pairs.add(pair)
    for (u, v) in pairs:
        w_uv = G[u][v]["weight"]
        w_vu = G[v][u]["weight"]
        if w_uv >= 2 and w_vu >= 2:
            ababa_accounts.add(u)
            ababa_accounts.add(v)

    # unique counterparties
    out_partners = defaultdict(set)
    in_partners = defaultdict(set)
    for u, v in G.edges():
        out_partners[u].add(v)
        in_partners[v].add(u)

    # amounts in/out
    out_amount = defaultdict(float)
    in_amount = defaultdict(float)
    out_cnt = defaultdict(int)
    in_cnt = defaultdict(int)
    for _, r in txs.iterrows():
        u = int(r.sender_id); v = int(r.receiver_id)
        out_amount[u] += float(r.amount); in_amount[v] += float(r.amount)
        out_cnt[u] += 1; in_cnt[v] += 1

    # velocity: tx per day (approx using distinct days)
    days_active_out = defaultdict(set)
    days_active_in = defaultdict(set)
    for _, r in txs.iterrows():
        d = pd.Timestamp(r.timestamp).date()
        days_active_out[int(r.sender_id)].add(d)
        days_active_in[int(r.receiver_id)].add(d)

    # betweenness approximated for speed on larger graphs: sample
    try:
        betw = nx.betweenness_centrality(Gu, k=min(500, n), seed=42, normalized=True)
    except Exception:
        betw = {i: 0.0 for i in range(n)}

    rows = []
    for i in range(n):
        in_c = in_cnt.get(i, 0)
        out_c = out_cnt.get(i, 0)
        tot_c = in_c + out_c
        vel_out = out_c / max(len(days_active_out.get(i, {1})), 1)
        vel_in = in_c / max(len(days_active_in.get(i, {1})), 1)
        ia = in_amount.get(i, 0.0)
        oa = out_amount.get(i, 0.0)
        up = len(out_partners.get(i, set()))
        ip = len(in_partners.get(i, set()))
        rows.append({
            "account_id": i,
            "out_degree_weighted": out_deg.get(i, 0),
            "in_degree_weighted": in_deg.get(i, 0),
            "in_degree": in_deg_u.get(i, 0),
            "out_degree": out_deg_u.get(i, 0),
            "total_degree": total_deg.get(i, 0),
            "pagerank": pr.get(i, 0.0),
            "clustering_coeff": clustering.get(i, 0.0),
            "connected_component_size": comp_size.get(i, 1),
            "community_id": comm_id.get(i, -1),
            "community_size": comm_size.get(i, 1),
            "betweenness": betw.get(i, 0.0),
            "unique_out_partners": up,
            "unique_in_partners": ip,
            "partner_diversity": (up + ip) / max(tot_c, 1),
            "out_amount": oa,
            "in_amount": ia,
            "net_amount": ia - oa,
            "avg_out_amount": oa / max(out_c, 1),
            "avg_in_amount": ia / max(in_c, 1),
            "in_out_ratio": ia / max(oa, 1e-6),
            "velocity_out": vel_out,
            "velocity_in": vel_in,
            "ababa_flag": int(i in ababa_accounts),
        })
    gdf = pd.DataFrame(rows)
    return gdf, G


def main():
    cfg = load_config()
    accounts = pd.read_parquet("data/accounts_raw.parquet")
    txs = pd.read_parquet("data/transactions.parquet")
    gdf, G = compute_graph_features(accounts, txs)
    # merge with account raw
    df = accounts.merge(gdf, on="account_id", how="left").fillna(0)
    # high-risk interaction features
    df["risk_country_x_chargebacks"] = df["country_risk"] * (df["prior_chargebacks"] + 1)
    df["low_trust_new_account"] = ((df["device_trust_score"] < 0.3) & (df["account_age_days"] < 90)).astype(int)

    out = cfg["data"]["output_path"]
    df.to_parquet(out, index=False)
    df.to_csv(out.replace(".parquet", ".csv"), index=False)
    # save adjacency
    import pickle
    with open("data/graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("Feature dataset:", df.shape, "fraud:", df["is_fraud"].sum())
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
