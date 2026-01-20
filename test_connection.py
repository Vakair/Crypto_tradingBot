import ccxt
import os
from dotenv import load_dotenv

# Betöltjük a jelszavakat a .env fájlból
load_dotenv()


def main():
    print(" KAPCSOLÓDÁS A BINANCE TESTNETHEZ...")

    # API Kulcsok ellenőrzése
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')

    if not api_key or not secret_key:
        print("!!! HIBA: Nem találom a kulcsokat a .env fájlban! !!!")
        return

    # Testnet beállítása
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # Egyelőre Spot (sima) kereskedés
        }
    })

    # Átállítjuk a Testnet URL-ekre (hogy ne az igazi pénzt piszkálja!)
    exchange.set_sandbox_mode(True)

    try:
        #Lekérjük az egyenleget
        print("   Egyenleg lekérdezése...")
        balance = exchange.fetch_balance()

        # Megkeressük, miből mennyi van (Free = elkölthető)
        usdt_balance = balance['USDT']['free']
        btc_balance = balance['BTC']['free']

        print(f"\n SIKERES KAPCSOLAT!")
        print(f"    Játék USDT Egyenleg: ${usdt_balance:,.2f}")
        print(f"    Játék BTC Egyenleg:  {btc_balance:.6f} BTC")

        #Lekérjük a BTC árfolyamot is
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"    Aktuális BTC ár: ${ticker['last']:,.2f}")

    except Exception as e:
        print(f"\n!!! KAPCSOLÓDÁSI HIBA: {e} !!!!")
        print("!!!   Ellenőrizd, hogy helyesek-e a kulcsok a .env fájlban! !!!")



if __name__ == "__main__":
    main()