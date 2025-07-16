# 🏦 Wallet Credit Scoring System

This project builds a **machine learning-inspired scoring pipeline** that assigns a **credit score (0–1000)** to DeFi wallets based on their **on-chain transaction behavior** using data from the **Aave V2 protocol** on the Polygon network.

---

## 📌 Objective

Given a dataset of 100,000 DeFi transactions, assign each wallet a score that reflects its **financial reliability**:

- **Higher scores (closer to 1000)** → responsible, human-like, reliable usage.
- **Lower scores (closer to 0)** → risky, bot-like, exploitative, or negligent behavior.

---

## 🧠 Methodology

We use a **rule-based scoring system** with engineered behavioral features from historical transaction data. This avoids requiring labeled data and reflects real-world heuristics used in DeFi risk analysis.

### Key Steps:

1. **Load & Preprocess JSON Data**
2. **Feature Engineering** (wallet-level aggregates)
3. **Scoring Based on Feature Weights**
4. **Export Scores & Analysis**

---

## 🏗️ Architecture

```text
user-wallet-transactions.json (raw data)
        │
        ▼
     scorer.py
        ├── Load & Flatten Transaction Data
        ├── Aggregate Wallet-Level Behavior
        ├── Engineer Features:
        │     • Total Borrowed, Deposited, Repaid
        │     • Repayment Ratios
        │     • Liquidation Count
        │     • Activity Frequency
        │     • Asset Diversity
        ├── Normalize & Score Wallets (0–1000)
        ├── Export:
        │     • wallet_scores.csv
        │     • analysis.md + score_distribution.png

---
```
## ⚙️ Feature List

| Feature               | Description                             |
| --------------------- | --------------------------------------- |
| `total_usd_deposited` | Total USD value of all deposits         |
| `total_usd_borrowed`  | Total USD value of borrowed funds       |
| `total_usd_repaid`    | Total USD repaid by wallet              |
| `repayment_ratio`     | Repaid / Borrowed                       |
| `borrow_to_deposit`   | Borrowed / Deposited                    |
| `num_liquidations`    | Number of liquidation calls             |
| `tx_count`            | Total number of transactions            |
| `active_days`         | Number of unique days active            |
| `asset_diversity`     | Number of unique tokens interacted with |

---

## 🔁 Processing Flow

1. Read JSON data and extract key fields
2. Group by wallet and compute metrics
3. Normalize features (e.g., min-max scaling)
4. Apply heuristic weights to features:
   - Positive Influence: repayment ratio, active days, diversity
   - Negative Influence: high borrow ratio, liquidations
5. Generate final score between 0 and 1000
6. Export results to CSV and analysis with score graph

---

## 📦 Output

- `output/wallet_scores.csv` → final scores
- `output/score_distribution.png` → visual distribution of scores
- `analysis.md` → descriptive analysis of low/high scoring behaviors

---

## 🚀 Requirements

```bash
pip install pandas matplotlib seaborn
```
