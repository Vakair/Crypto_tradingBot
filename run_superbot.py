import time
import json
import os
from datetime import datetime
from src.strategy_manager import StrategyManager

CONFIG_FILE = 'strategies.json'


def load_config():
    """Betölti, hogy melyik stratégia van bekapcsolva"""
    if not os.path.exists(CONFIG_FILE):
        return {"swing": False, "daytrade": False, "pair": False}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"swing": False, "daytrade": False, "pair": False}


def main():
    print("==========================================")
    print(" SUPER BOT - TÁVIRÁNYÍTOTT MÓD")
    print("==========================================")
    #print("Várakozás a parancsokra a Webes Appból...")

    # Manager Inicializálása
    bot = StrategyManager()

    #Végtelen Ciklus
    while True:
        try:
            # Minden körben megnézzük, mit kapcsolt be a felhasználó
            config = load_config()

            active_strategies = [k for k, v in config.items() if v]
            timestamp = datetime.now().strftime('%H:%M:%S')

            print(f"\n {timestamp} | Aktív: {active_strategies}")

            # --- SWING STRATÉGIA ---
            if config.get('swing'):
                print("    Swing Trade futtatása...")
                status = bot.run_swing_strategy()
                print(f"      -> {status}")

            # --- DAYTRADE STRATÉGIA ---
            if config.get('daytrade'):
                print("    Daytrade futtatása...")
                status = bot.run_daytrade_strategy()
                print(f"      -> {status}")

            # --- PAIR TRADING ---
            if config.get('pair'):
                print("    Pair Trading futtatása...")
                status = bot.run_pair_strategy()
                print(f"      -> {status}")

            if not active_strategies:
                print("   💤 Minden stratégia PIHEN. (Kapcsold be az Appban!)")

            # Várakozás (Élesben 1 perc, most tesztre 10 mp)
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n!!! Bot leállítva. !!!")
            break
        except Exception as e:
            print(f"\n!!! Hiba történt: {e} !!!")
            time.sleep(5)


if __name__ == "__main__":
    main()