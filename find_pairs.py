import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import numpy as np

# --- KONFIGURÁCIÓ ---
# Ezeket a coinokat vizsgáljuk meg (Yahoo Finance kódok)
TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'LINK-USD', 'SUI-USD', 'DOT-USD']
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"


def download_data():
    print("Adatok letöltése az elemzéshez...")
    df = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
    df = df.dropna()
    print(f"Letöltve: {len(df)} nap adata.")
    return df


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []

    print("\n🔍 Kointegráció vizsgálata (P-értékek számítása)...")
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]

            # Kointegrációs teszt (Engle-Granger)
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]

            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            # Ha p < 0.05, akkor statisztikailag szignifikáns a kapcsolat!
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))
                print(f"   ✨ TALÁLAT! {keys[i]} - {keys[j]} (p={pvalue:.4f})")

    return pvalue_matrix, pairs, keys





def main():
    #Adat
    data = download_data()

    #Elemzés
    pvalues, pairs, keys = find_cointegrated_pairs(data)

    #Hőtérkép
    plt.figure(figsize=(10, 8))
    sns.heatmap(pvalues, xticklabels=keys, yticklabels=keys, cmap='RdYlGn_r', mask=(pvalues >= 0.99))
    plt.title('Kointegrációs P-értékek (Zöld = Jó Pár)')
    plt.tight_layout()
    plt.savefig('results/pair_heatmap.png')
    print("\nHőtérkép mentve: results/pair_heatmap.png")

    # 4. Legjobb pár kiválasztása
    print("\n LEGJOBB PÁROK (p < 0.05):")
    print("-" * 30)
    if not pairs:
        print("Nincs tökéletes pár")
    else:
        # Rendezzük p-érték szerint (a legkisebb a legerősebb kapcsolat)
        pairs.sort(key=lambda x: x[2])
        for p in pairs:
            print(f"1. {p[0]} - {p[1]} \t(p={p[2]:.5f})")

        best_pair = pairs[0]
        print("-" * 30)
        print(f" Legerősebb pár: {best_pair[0]} és {best_pair[1]}")


if __name__ == "__main__":
    main()