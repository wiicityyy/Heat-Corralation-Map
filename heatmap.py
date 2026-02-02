import os
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram


# ----------------------------
# Config (edit these)
# ----------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD"
]
START_DATE = "2019-01-01"

WINDOW_SHORT = 60     # ~3 months of trading days
WINDOW_LONG = 252     # ~1 year of trading days

OUTDIR = "outputs"
HEATMAP_FILE = os.path.join(OUTDIR, "heatmap_clustered.png")
HEATMAP_2WIN_FILE = os.path.join(OUTDIR, "heatmap_60_vs_252.png")
DENDRO_FILE = os.path.join(OUTDIR, "dendrogram.png")
CORR_CSV_FILE = os.path.join(OUTDIR, "corr_matrix.csv")
PAIRS_TXT_FILE = os.path.join(OUTDIR, "pairs.txt")


# ----------------------------
# Helpers
# ----------------------------
def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=True
    )["Close"]
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices).diff().dropna()


def corr_last_n(returns: pd.DataFrame, n: int) -> pd.DataFrame:
    """Correlation matrix using the last n rows."""
    if len(returns) < n:
        raise ValueError(f"Not enough data for window {n}. Have {len(returns)} rows.")
    return returns.tail(n).corr()


def flatten_corr_pairs(corr: pd.DataFrame) -> pd.Series:
    """Return upper-triangle correlation pairs as a sorted Series."""
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)
    return pairs


def avg_abs_corr(corr: pd.DataFrame) -> float:
    """Average absolute correlation excluding diagonal."""
    mask_diag = ~np.eye(len(corr), dtype=bool)
    return corr.where(mask_diag).abs().stack().mean()


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Plots
# ----------------------------
def plot_clustered_heatmap(corr: pd.DataFrame, title: str, save_path: str) -> None:
    """(1) Clustered heatmap + upper-triangle mask."""
    # Linkage on correlation (note: corr is similarity; linkage expects distance-like,
    # but for visualization this is fine; for strictness you could use (1-corr)/2.)
    Z = linkage(corr, method="average")
    order = leaves_list(Z)
    corr_sorted = corr.iloc[order, order]

    # Mask the upper triangle for readability
    mask = np.triu(np.ones_like(corr_sorted, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_sorted,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_two_window_heatmaps(corr_60: pd.DataFrame, corr_252: pd.DataFrame, save_path: str) -> None:
    """(2) Two-window heatmaps (60 vs 252)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        corr_60, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.5, square=True, ax=axes[0], cbar=False
    )
    axes[0].set_title("Correlation (Last 60 trading days)")

    sns.heatmap(
        corr_252, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.5, square=True, ax=axes[1], cbar=True
    )
    axes[1].set_title("Correlation (Last 252 trading days)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_dendrogram(corr: pd.DataFrame, save_path: str) -> None:
    """(6) Dendrogram from clustering."""
    Z = linkage(corr, method="average")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=corr.columns, leaf_rotation=45)
    plt.title("Correlation Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_outdir(OUTDIR)

    # Data
    prices = download_prices(TICKERS, START_DATE)
    if prices.empty or prices.shape[1] < 2:
        raise RuntimeError("Not enough price data downloaded. Try different tickers or a different date range.")

    returns = log_returns(prices)

    corr = returns.corr()

    corr.to_csv(CORR_CSV_FILE)

    plot_clustered_heatmap(
        corr,
        title="Clustered Correlation Heatmap (Daily Returns)",
        save_path=HEATMAP_FILE
    )

    corr_60 = corr_last_n(returns, WINDOW_SHORT)
    corr_252 = corr_last_n(returns, WINDOW_LONG)
    plot_two_window_heatmaps(corr_60, corr_252, HEATMAP_2WIN_FILE)

    pairs = flatten_corr_pairs(corr)

    top5 = pairs.head(5)
    bottom5 = pairs.tail(5)

    div_score = avg_abs_corr(corr)

    print("\nTop 5 most correlated pairs:")
    print(top5)

    print("\nTop 5 least correlated pairs:")
    print(bottom5)

    print(f"\nAverage absolute correlation (diversification score): {div_score:.2f}")
    print(f"\nSaved outputs in: {OUTDIR}/")

    with open(PAIRS_TXT_FILE, "w") as f:
        f.write("Top 10 correlated pairs:\n")
        f.write(str(pairs.head(10)))
        f.write("\n\nBottom 10 correlated pairs:\n")
        f.write(str(pairs.tail(10)))
        f.write(f"\n\nAverage absolute correlation (diversification score): {div_score:.4f}\n")

    # Dendrogram
    plot_dendrogram(corr, DENDRO_FILE)


if __name__ == "__main__":
    main()
