import pandas as pd
import matplotlib.pyplot as plt

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
TARGET_SYMBOL = 'BTCUSDT'
START_DATE = '2022-01-01'
END_DATE = '2022-07-01'


def main():
    print(f"ADAT DEBUGGOLÁS ({TARGET_SYMBOL})...")

    # Betöltés
    df = pd.read_csv(DATA_PATH)

    # Szűrés
    df = df[df['symbol'] == TARGET_SYMBOL].copy()

    #Dátum konvertálás és rendezés
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    #Időszak kivágása
    mask = (df.index >= START_DATE) & (df.index <= END_DATE)
    df_slice = df.loc[mask]

    if df_slice.empty:
        print("HIBA: Üres az adathalmaz ebben az időszakban!")
        return

    #Elemzés
    start_price = df_slice.iloc[0]['close']
    end_price = df_slice.iloc[-1]['close']
    price_change = ((end_price - start_price) / start_price) * 100


    print(f"   Első nap ({df_slice.index[0].date()}): ${start_price:,.2f}")
    print(f"   Utolsó nap ({df_slice.index[-1].date()}): ${end_price:,.2f}")
    print(f"   Változás: {price_change:+.2f}%")
    print(f"   Adatsorok száma: {len(df_slice)}")

    #Duplikáció ellenőrzés
    # Vannak-e azonos dátumok? (Ez okozhat zavart)
    duplicates = df_slice.index.duplicated().sum()
    print(f"   Duplikált dátumok száma: {duplicates}")

    #Kirajzolás
    plt.figure(figsize=(12, 6))
    plt.plot(df_slice.index, df_slice['close'], label='Close Price', color='blue')
    plt.title(f'Tényleges Árfolyam: {START_DATE} - {END_DATE}')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()