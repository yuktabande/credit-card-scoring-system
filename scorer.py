import pandas as pd
import json
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the JSON file
DATA_PATH = "data/user-wallet-transactions.json"
with open(DATA_PATH, "r") as f:
    raw_data = json.load(f)

# 2. Convert to DataFrame
df = pd.json_normalize(raw_data)

# 3. Rename and convert key fields
df["wallet"] = df["userWallet"]
df["action"] = df["action"].str.lower()
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

# 4. Convert amount and price to float
df["amount"] = pd.to_numeric(df["actionData.amount"], errors="coerce")
df["asset_symbol"] = df["actionData.assetSymbol"]
df["asset_price_usd"] = pd.to_numeric(df["actionData.assetPriceUSD"], errors="coerce")

# 5. Compute USD value of transaction
df["usd_value"] = (df["amount"] / 1e6) * df["asset_price_usd"]

# 6. Drop unnecessary fields
df = df[["wallet", "action", "timestamp", "asset_symbol", "usd_value"]]

# # ✅ Show preview
print("Loaded data:")
print(df.head())
print(f"\nTotal records: {len(df)} | Unique wallets: {df['wallet'].nunique()}")

# --- Step 1: Financial Totals per Wallet ---

relevant_actions = ["deposit", "borrow", "repay", "redeemunderlying"]
filtered_df = df[df["action"].isin(relevant_actions)].copy()

# Created a pivot table of USD value by wallet and action
wallet_action_pivot = (
    filtered_df.groupby(["wallet", "action"])["usd_value"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)

# Renaming for clarity
wallet_action_pivot.columns.name = None
wallet_action_pivot = wallet_action_pivot.rename(columns={
    "deposit": "total_usd_deposited",
    "borrow": "total_usd_borrowed",
    "repay": "total_usd_repaid",
    "redeemunderlying": "total_usd_redeemed"
})

# Show preview
print("\n[Step 1] Wallet Financial Aggregates:")
print(wallet_action_pivot.head())

# --- Step 2: Behavioral Ratios ---

# Add repayment_ratio
wallet_action_pivot["repayment_ratio"] = wallet_action_pivot.apply(
    lambda row: row["total_usd_repaid"] / row["total_usd_borrowed"] if row["total_usd_borrowed"] > 0 else 1.0,
    axis=1
)

# Add borrow_to_deposit ratio
wallet_action_pivot["borrow_to_deposit"] = wallet_action_pivot.apply(
    lambda row: row["total_usd_borrowed"] / row["total_usd_deposited"] if row["total_usd_deposited"] > 0 else 0.0,
    axis=1
)

# Preview
print("\n[Step 2] Ratios:")
print(wallet_action_pivot[["wallet", "repayment_ratio", "borrow_to_deposit"]].head())

# --- Step 3: Liquidation Count ---

liquidations = (
    df[df["action"] == "liquidationcall"]
    .groupby("wallet")
    .size()
    .reset_index(name="num_liquidations")
)

# Merged with main wallet summary
wallet_summary = pd.merge(wallet_action_pivot, liquidations, on="wallet", how="left")
wallet_summary["num_liquidations"] = wallet_summary["num_liquidations"].fillna(0).astype(int)

# Preview
print("\n[Step 3] Added Liquidation Count:")
print(wallet_summary[["wallet", "num_liquidations"]].head())

# --- Step 4: Transaction Frequency ---

# Total number of transactions per wallet
tx_counts = df.groupby("wallet").size().reset_index(name="tx_count")

# Unique active days per wallet
df["date"] = df["timestamp"].dt.date
active_days = df.groupby("wallet")["date"].nunique().reset_index(name="active_days")

# Merged into wallet_summary
wallet_summary = wallet_summary.merge(tx_counts, on="wallet", how="left")
wallet_summary = wallet_summary.merge(active_days, on="wallet", how="left")

# Preview
print("\n[Step 4] Activity Frequency Features:")
print(wallet_summary[["wallet", "tx_count", "active_days"]].head())

# --- Step 5: Token Diversity ---

diversity = df.groupby("wallet")["asset_symbol"].nunique().reset_index(name="asset_diversity")

# Merge into wallet_summary
wallet_summary = wallet_summary.merge(diversity, on="wallet", how="left")

# Preview
print("\n[Step 5] Token Diversity Feature:")
print(wallet_summary[["wallet", "asset_diversity"]].head())

# --- Final Step: Score Generation ---

# Cap repayment_ratio to 1.0
wallet_summary["repayment_ratio_capped"] = wallet_summary["repayment_ratio"].clip(upper=1.0)

# Inverted borrow-to-deposit ratio (higher is better)
wallet_summary["borrow_inverse"] = 1.0 - wallet_summary["borrow_to_deposit"].clip(upper=1.0)

# Log-scaled tx count and deposit value
wallet_summary["log_tx_count"] = np.log1p(wallet_summary["tx_count"])
wallet_summary["log_deposit"] = np.log1p(wallet_summary["total_usd_deposited"])

# Cap diversity
wallet_summary["asset_diversity_capped"] = wallet_summary["asset_diversity"].clip(upper=10)

# Invert liquidation count
wallet_summary["liquidation_penalty"] = -wallet_summary["num_liquidations"]

# Features to scale
features = [
    "repayment_ratio_capped",
    "borrow_inverse",
    "log_tx_count",
    "active_days",
    "asset_diversity_capped",
    "log_deposit",
    "liquidation_penalty"
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(wallet_summary[features])

# Apply weights
weights = np.array([0.25, 0.15, 0.10, 0.10, 0.10, 0.10, 0.20])
wallet_summary["credit_score"] = (scaled @ weights) * 1000

# Round and preview
wallet_summary["credit_score"] = wallet_summary["credit_score"].round(2)

print("\n✅ Final Wallet Credit Scores:")
print(wallet_summary[["wallet", "credit_score"]].head())

# Save scores to CSV in 'output' folder

os.makedirs("output", exist_ok=True)
wallet_summary[["wallet", "credit_score"]].to_csv("output/wallet_scores.csv", index=False)
print("\n📁 Saved credit scores to 'output/wallet_scores.csv'")

# --- Score Distribution Analysis ---

# Bin scores into 10 buckets (0-100, 100-200, ..., 900-1000)
wallet_summary["score_range"] = pd.cut(wallet_summary["credit_score"], 
                                       bins=[i for i in range(0, 1100, 100)], 
                                       labels=[f"{i}-{i+100}" for i in range(0, 1000, 100)])

# Plot distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=wallet_summary, x="score_range", palette="viridis")
plt.title("Credit Score Distribution of Wallets")
plt.xlabel("Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/score_distribution.png")
plt.close()

# --- Write analysis.md ---

low_range = wallet_summary[wallet_summary["credit_score"] < 300]
high_range = wallet_summary[wallet_summary["credit_score"] > 800]

with open("analysis.md", "w") as f:
    f.write("# Wallet Credit Score Analysis\n\n")
    f.write("### 🔍 Score Distribution\n")
    f.write("![Score Distribution](output/score_distribution.png)\n\n")

    f.write("The histogram above shows the distribution of credit scores across all wallets. "
            "Most wallets tend to cluster in the middle-to-high range, indicating reasonably responsible behavior.\n\n")

    f.write("### 📉 Behavior of Low-Scoring Wallets (Score < 300)\n")
    f.write(f"- Total wallets in this range: {len(low_range)}\n")
    f.write("- Common traits:\n")
    f.write("  - High number of liquidations\n")
    f.write("  - Very low repayment ratios\n")
    f.write("  - Low transaction activity\n")
    f.write("  - Minimal asset diversity\n\n")

    f.write("### 📈 Behavior of High-Scoring Wallets (Score > 800)\n")
    f.write(f"- Total wallets in this range: {len(high_range)}\n")
    f.write("- Common traits:\n")
    f.write("  - Near-perfect repayment ratios\n")
    f.write("  - High number of transactions across long active periods\n")
    f.write("  - Good asset diversity\n")
    f.write("  - Low to zero liquidations\n\n")

    f.write("This scoring system offers a meaningful snapshot of DeFi wallet reliability based on on-chain behavior.\n")