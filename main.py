import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

DATA_DIR = "tsla2019"

MESSAGE_COLS = ["time", "type", "order_id", "size", "price", "direction", "trade_price"]
MESSAGE_USECOLS = [0, 1, 2, 3, 4, 5]  # exclude trade_price
ORDERBOOK_COLS = [
    "ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1",
    "ask_price_2", "ask_size_2", "bid_price_2", "bid_size_2",
]
ORDERBOOK_USECOLS = [0, 1, 2, 3]  # ask_price_1, ask_size_1, bid_price_1, bid_size_1

def load_day(message_path: str, orderbook_path: str) -> pd.DataFrame:
    msg = pd.read_csv(message_path, header=None, names=MESSAGE_COLS, usecols=MESSAGE_USECOLS, na_values="null", low_memory=False)
    ob  = pd.read_csv(orderbook_path, header=None, names=ORDERBOOK_COLS[:4], usecols=ORDERBOOK_USECOLS)
    df = pd.concat([msg, ob], axis=1)
    df = df[df["type"].isin([4, 5])].reset_index(drop=True)

    # extract date from filename: TSLA_2019-01-02_...
    date_str = os.path.basename(message_path).split("_")[1]
    date = pd.Timestamp(date_str)
    df["time"] = date + pd.to_timedelta(df["time"], unit="s")

    for col in ["price", "ask_price_1", "bid_price_1"]:
        df[col] = df[col] / 10000

    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2

    return df

def load_all() -> pd.DataFrame:
    message_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_message_2.csv")))
    days = []
    for msg_path in message_files:
        ob_path = msg_path.replace("_message_2.csv", "_orderbook_2.csv")
        if not os.path.exists(ob_path):
            print(f"Missing orderbook for {msg_path}, skipping.")
            continue
        days.append(load_day(msg_path, ob_path))
    return pd.concat(days, ignore_index=True)

def plot(df: pd.DataFrame):
    # resample par minute pour éviter 38M points
    df_indexed = df.set_index("time")
    mid = df_indexed["mid_price"].resample("1min").last().dropna()
    spread = df_indexed["spread"].resample("1min").mean().dropna()

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(mid.index, mid.values, linewidth=0.8)
    ax1.set_ylabel("Mid-price ($)")
    ax1.set_title("TSLA 2019 — Mid-price")

    ax2.plot(spread.index, spread.values, linewidth=0.8, color="orange")
    ax2.set_ylabel("Spread ($)")
    ax2.set_title("TSLA 2019 — Spread (ask - bid)")

    plt.tight_layout()
    plt.savefig("tsla_2019.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    df = load_all()
    print(df.shape)
    print(df.head())
    plot(df)

