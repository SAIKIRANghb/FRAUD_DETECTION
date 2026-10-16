"""Synthetic account + transaction generator with injected fraud rings.

Produces an account-level dataset enriched with raw tabular attributes.
Graph features are computed separately (see src/features/graph_features.py).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, set_seed


def generate_accounts(n, rng):
    countries = np.array(["US", "GB", "DE", "NG", "RU", "CN", "BR", "IN"])
    # higher-risk countries
    country_risk = {"US": 0.05, "GB": 0.06, "DE": 0.05, "NG": 0.35,
                    "RU": 0.40, "CN": 0.25, "BR": 0.20, "IN": 0.15}
    country = rng.choice(countries, size=n, p=[0.45, 0.1, 0.1, 0.08, 0.07, 0.07, 0.06, 0.07])
    account_age = rng.exponential(scale=400, size=n).clip(1, 3650).astype(int)
    kyc_verified = (rng.random(n) < 0.75).astype(int)
    prior_chargebacks = rng.poisson(0.3, size=n)
    device_trust = rng.beta(4, 2, size=n)
    email_age_days = rng.exponential(scale=800, size=n).clip(1, 5000).astype(int)
    avg_ticket = rng.lognormal(mean=3.5, sigma=1.0, size=n)
    return pd.DataFrame({
        "account_id": np.arange(n),
        "country": country,
        "country_risk": [country_risk[c] for c in country],
        "account_age_days": account_age,
        "kyc_verified": kyc_verified,
        "prior_chargebacks": prior_chargebacks,
        "device_trust_score": device_trust,
        "email_age_days": email_age_days,
        "avg_ticket_size": avg_ticket,
    })


def generate_fraud_rings(accounts, rng, fraud_rate):
    n = len(accounts)
    n_fraud = int(n * fraud_rate)
    # pick fraud ring accounts (concentrated)
    candidates = rng.choice(n, size=n_fraud, replace=False)
    fraud_set = set(candidates.tolist())
    return fraud_set


def generate_transactions(accounts_df, fraud_set, n_tx, rng):
    n_acct = len(accounts_df)
    # senders
    sender = rng.integers(0, n_acct, size=n_tx)
    # receivers biased: normal random; fraud ring members send to each other
    receiver = rng.integers(0, n_acct, size=n_tx)
    # inject ring structure: for some tx, if sender fraud, receiver likely fraud
    mask_fraud_sender = np.array([s in fraud_set for s in sender])
    n_inject = mask_fraud_sender.sum()
    # make receivers among fraud set for injected
    receiver[mask_fraud_sender] = rng.choice(list(fraud_set), size=n_inject, replace=True)

    amount = np.round(rng.lognormal(mean=3.2, sigma=1.4, size=n_tx), 2)
    # fraud transactions tend to be rounder / higher
    amount = np.where(mask_fraud_sender, np.round(amount * 1.3, 2), amount)
    amount = np.clip(amount, 1, 5000)

    channel = rng.choice(["web", "mobile", "api"], size=n_tx, p=[0.4, 0.5, 0.1])
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 90, size=n_tx), unit="D")
    is_fraud_tx = mask_fraud_sender & (rng.random(n_tx) < 0.6)

    df = pd.DataFrame({
        "tx_id": np.arange(n_tx),
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount,
        "channel": channel,
        "timestamp": ts,
        "is_fraud_tx": is_fraud_tx.astype(int),
    })
    # account-level label: fraud if in fraud_set
    accounts_df = accounts_df.copy()
    accounts_df["is_fraud"] = accounts_df["account_id"].map(lambda a: int(a in fraud_set))
    return df, accounts_df


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["data"]["n_accounts"]
    n_tx = cfg["data"]["n_transactions"]
    fraud_rate = cfg["data"]["fraud_rate"]

    accounts = generate_accounts(n, rng)
    fraud_set = generate_fraud_rings(accounts, rng, fraud_rate)
    txs, accounts = generate_transactions(accounts, fraud_set, n_tx, rng)

    os.makedirs("data", exist_ok=True)
    txs.to_parquet("data/transactions.parquet", index=False)
    accounts.to_parquet("data/accounts_raw.parquet", index=False)
    accounts.to_csv("data/accounts_raw.csv", index=False)
    txs.to_csv("data/transactions.csv", index=False)
    print("Generated accounts:", len(accounts), "fraud:", accounts["is_fraud"].sum())
    print("Generated transactions:", len(txs), "fraud tx:", txs["is_fraud_tx"].sum())


if __name__ == "__main__":
    main()
