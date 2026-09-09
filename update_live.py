from __future__ import annotations
import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "live_state.json"
OUT_FILE = ROOT / "live_signals.json"

WITA = timezone(timedelta(hours=8))
BUY_FEE = 0.0015
SELL_FEE = 0.0015
INITIAL_CAPITAL = 10_000_000.0

TICKERS = ["AADI", "AALI", "ABMM", "ACES", "ADMG", "ADMR", "ADRO", "AGAR", "AGII", "AISA", "AKRA", "ALDO", "ALKA", "ANTM", "APLN", "ARCI", "ARII", "ARNA", "ASGR", "ASLI", "ASRI", "ASSA", "ATIC", "AUTO", "AVIA", "AYAM", "BANK", "BBRM", "BELI", "BESS", "BEST", "BIRD", "BISI", "BKDP", "BKSL", "BLES", "BLOG", "BLUE", "BMHS", "BMTR", "BOGA", "BRIS", "BRMS", "BSDE", "BSSR", "BTPS", "BUAH", "BUDI", "BULL", "BUMI", "BWPT", "BYAN", "CAMP", "CARE", "CASS", "CITA", "CLEO", "CMNP", "CMRY", "CPIN", "CSAP", "CSRA", "CTRA", "DATA", "DEPO", "DEWA", "DGWG", "DILD", "DKFT", "DMAS", "DMMX", "DOOH", "DRMA", "DSNG", "DWGL", "EKAD", "ELPI", "ELSA", "ENRG", "ERAA", "ERAL", "ESSA", "EXCL", "FAST", "FILM", "FORE", "FPNI", "GDST", "GGRP", "GIAA", "GJTL", "GOLF", "GOOD", "GRIA", "GWSA", "GZCO", "HATM", "HEAL", "HERO", "HEXA", "HRTA", "HRUM", "IATA", "ICBP", "IFII", "IFSH", "IMPC", "INDF", "INDS", "INDY", "INET", "INKP", "INPP", "INTP", "IPCM", "IRSX", "ISAT", "ISSP", "ITMA", "ITMG", "JARR", "JAWA", "JGLE", "JIHD", "JKON", "JPFA", "JRPT", "JSMR", "JTPE", "KBLI", "KEEN", "KEJU", "KETR", "KIJA", "KKGI", "KLBF", "KOTA", "KPIG", "LPCK", "LPPF", "LSIP", "LTLS", "MAHA", "MAIN", "MAPA", "MAPI", "MARK", "MBAP", "MBMA", "MDIA", "MDIY", "MDKA", "MDLA", "MEDC", "MIKA", "MINE", "MKAP", "MKTR", "MLIA", "MLPL", "MMIX", "MMLP", "MNCN", "MORA", "MPMX", "MSIN", "MSJA", "MSTI", "MTDL", "MTEL", "MYOH", "MYOR", "NEST", "NICE", "NICL", "NRCA", "NSSS", "OMED", "PALM", "PBID", "PGAS", "PKPK", "PMJS", "POWR", "PPRE", "PRDA", "PSAB", "PSAT", "PSGO", "PSKT", "PSSI", "PTBA", "PTPP", "PTPW", "PTSN", "RAAM", "RALS", "RATU", "RISE", "ROCK", "ROTI", "RSCH", "SAME", "SAMF", "SGER", "SGRO", "SIDO", "SILO", "SIMP", "SKLT", "SKRN", "SMAR", "SMBR", "SMDM", "SMDR", "SMGR", "SMIL", "SMMT", "SMRA", "SMSM", "SOCI", "SPTO", "SRTG", "SSIA", "STAA", "SUNI", "TAPG", "TCPI", "TEBE", "TINS", "TKIM", "TLDN", "TLKM", "TMAS", "TOBA", "TOTL", "TOTO", "TPIA", "TPMA", "TRST", "TSPC", "UANG", "UCID", "ULTJ", "UNTR", "UNVR", "VERN", "VICI", "VISI", "VKTR", "WIFI", "WOOD", "YUPI"]
VARIANTS = ["D_TREND_MA20_GT_MA50", "A_BASELINE_MA50", "C_EARLY_BUY_MA20_EXIT_MA50", "B_MA20_FULL"]

# Exact rule map from the 4-variant backtest
VARIANT_RULES = {
    "A_BASELINE_MA50": {
        "buy": "CLOSE_GT_MA50",
        "exit": "CLOSE_LT_MA50",
    },
    "B_MA20_FULL": {
        "buy": "CLOSE_GT_MA20",
        "exit": "CLOSE_LT_MA20",
    },
    "C_EARLY_BUY_MA20_EXIT_MA50": {
        "buy": "CLOSE_GT_MA20",
        "exit": "CLOSE_LT_MA50",
    },
    "D_TREND_MA20_GT_MA50": {
        "buy": "CLOSE_GT_MA20_AND_MA20_GT_MA50",
        "exit": "CLOSE_LT_MA50",
    },
}

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100).where(avg_loss != 0, 100)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["EMA8"] = x["Close"].ewm(span=8, adjust=False).mean()
    x["EMA21"] = x["Close"].ewm(span=21, adjust=False).mean()
    x["MACD"] = x["EMA8"] - x["EMA21"]
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=5, adjust=False).mean()
    x["MACD_GC"] = (x["MACD"] > x["MACD_SIGNAL"]) & (x["MACD"].shift(1) <= x["MACD_SIGNAL"].shift(1))
    x["MACD_DC"] = (x["MACD"] < x["MACD_SIGNAL"]) & (x["MACD"].shift(1) >= x["MACD_SIGNAL"].shift(1))
    x["MA20"] = x["Close"].rolling(20, min_periods=20).mean()
    x["MA50"] = x["Close"].rolling(50, min_periods=50).mean()
    x["RSI14"] = rsi_wilder(x["Close"], 14)
    x["VOL_MA20"] = x["Volume"].rolling(20, min_periods=20).mean()
    x["VOL_OK"] = x["Volume"] > x["VOL_MA20"]
    return x

def buy_signal(row: pd.Series, variant: str) -> bool:
    common = (
        bool(row["MACD_GC"])
        and float(row["MACD"]) > 0
        and float(row["RSI14"]) > 50
        and bool(row["VOL_OK"])
    )
    if not common:
        return False
    rule = VARIANT_RULES[variant]["buy"]
    if rule == "CLOSE_GT_MA50":
        return float(row["Close"]) > float(row["MA50"])
    if rule == "CLOSE_GT_MA20":
        return float(row["Close"]) > float(row["MA20"])
    if rule == "CLOSE_GT_MA20_AND_MA20_GT_MA50":
        return (float(row["Close"]) > float(row["MA20"])) and (float(row["MA20"]) > float(row["MA50"]))
    return False

def exit_signal(row: pd.Series, variant: str) -> bool:
    if not bool(row["MACD_DC"]):
        return False
    rule = VARIANT_RULES[variant]["exit"]
    if rule == "CLOSE_LT_MA20":
        return float(row["Close"]) < float(row["MA20"])
    return float(row["Close"]) < float(row["MA50"])

def safe_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    state = {
        "version": 1,
        "startForwardDate": datetime.now(WITA).strftime("%Y-%m-%d"),
        "initialCapital": INITIAL_CAPITAL,
        "updatedAtWITA": None,
        "portfolios": {}
    }
    for t in TICKERS:
        state["portfolios"][t] = {}
        for v in VARIANTS:
            state["portfolios"][t][v] = {
                "cash": INITIAL_CAPITAL, "qty": 0.0, "costBasis": 0.0,
                "entryDate": None, "entryPrice": None,
                "pendingBuySignalDate": None, "pendingExitSignalDate": None,
                "trades": []
            }
    return state

def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker + ".JK",
            period="9mo",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        needed = ["Open","High","Low","Close","Volume"]
        if any(c not in df.columns for c in needed):
            return None
        x = df[needed].dropna(subset=["Open","Close"]).copy()
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return add_indicators(x)
    except Exception as e:
        print("DOWNLOAD_FAIL", ticker, repr(e))
        return None

def execute_pending(port, today_date, today_open, ticker, variant, pending_exec):
    # Pending EXIT gets priority because portfolio is currently open.
    if port["qty"] > 0 and port.get("pendingExitSignalDate"):
        signal_date = port["pendingExitSignalDate"]
        if today_date > signal_date:
            gross = port["qty"] * today_open
            proceeds = gross * (1 - SELL_FEE)
            pnl = proceeds - port["costBasis"]
            ret = (proceeds / port["costBasis"] - 1) * 100 if port["costBasis"] else 0.0
            trade = {
                "status": "CLOSED",
                "signalEntryDate": port["trades"][-1].get("signalEntryDate") if port["trades"] else None,
                "entryDate": port["entryDate"],
                "entryPrice": port["entryPrice"],
                "exitSignalDate": signal_date,
                "exitDate": today_date,
                "exitPrice": today_open,
                "qty": port["qty"],
                "netReturnPct": ret,
                "pnlIDR": pnl,
                "equityAfter": port["cash"] + proceeds,
            }
            # Replace currently open trade with closed form
            if port["trades"] and port["trades"][-1].get("status") == "OPEN":
                port["trades"][-1] = trade
            else:
                port["trades"].append(trade)
            port["cash"] += proceeds
            port["qty"] = 0.0
            port["costBasis"] = 0.0
            port["entryDate"] = None
            port["entryPrice"] = None
            port["pendingExitSignalDate"] = None
            pending_exec.append({"ticker":ticker,"variant":variant,"type":"EXIT_EXECUTED","date":today_date,"price":today_open})

    if port["qty"] == 0 and port.get("pendingBuySignalDate"):
        signal_date = port["pendingBuySignalDate"]
        if today_date > signal_date:
            budget = port["cash"] / (1 + BUY_FEE)
            qty = budget / today_open
            gross = qty * today_open
            fee = gross * BUY_FEE
            cost = gross + fee
            port["cash"] -= cost
            port["qty"] = qty
            port["costBasis"] = cost
            port["entryDate"] = today_date
            port["entryPrice"] = today_open
            port["trades"].append({
                "status": "OPEN",
                "signalEntryDate": signal_date,
                "entryDate": today_date,
                "entryPrice": today_open,
                "qty": qty,
                "costBasis": cost,
            })
            port["pendingBuySignalDate"] = None
            pending_exec.append({"ticker":ticker,"variant":variant,"type":"BUY_EXECUTED","date":today_date,"price":today_open})

def main():
    now = datetime.now(WITA)
    state = load_state()

    new_buy = []
    new_exit = []
    pending_exec = []
    positions = []
    snapshots = {}
    market_date_global = None

    for n, ticker in enumerate(TICKERS, 1):
        df = download_ticker(ticker)
        if df is None or len(df) < 60:
            continue
        row = df.iloc[-1]
        today = df.index[-1].strftime("%Y-%m-%d")
        if market_date_global is None or today > market_date_global:
            market_date_global = today
        today_open = float(row["Open"])
        today_close = float(row["Close"])

        snapshots[ticker] = {
            "date": today,
            "open": safe_float(row["Open"]),
            "close": safe_float(row["Close"]),
            "ma20": safe_float(row["MA20"]),
            "ma50": safe_float(row["MA50"]),
            "macd": safe_float(row["MACD"]),
            "macdSignal": safe_float(row["MACD_SIGNAL"]),
            "rsi14": safe_float(row["RSI14"]),
            "volume": safe_float(row["Volume"]),
            "volMA20": safe_float(row["VOL_MA20"]),
        }

        for variant in VARIANTS:
            port = state["portfolios"][ticker][variant]

            # First, execute any signal confirmed on a prior trading day at today's open.
            execute_pending(port, today, today_open, ticker, variant, pending_exec)

            # Then evaluate today's confirmed close for a NEW signal.
            # Avoid duplicate confirmation for the same date.
            if port["qty"] == 0 and not port.get("pendingBuySignalDate"):
                if buy_signal(row, variant):
                    port["pendingBuySignalDate"] = today
                    new_buy.append({
                        "ticker": ticker,
                        "variant": variant,
                        "signalDate": today,
                        "close": today_close,
                        "macd": safe_float(row["MACD"]),
                        "rsi14": safe_float(row["RSI14"]),
                    })
            elif port["qty"] > 0 and not port.get("pendingExitSignalDate"):
                if exit_signal(row, variant):
                    port["pendingExitSignalDate"] = today
                    new_exit.append({
                        "ticker": ticker,
                        "variant": variant,
                        "signalDate": today,
                        "close": today_close,
                        "entryDate": port.get("entryDate"),
                        "entryPrice": port.get("entryPrice"),
                    })

            if port["qty"] > 0:
                mtm = port["cash"] + port["qty"] * today_close
                unreal = (mtm / port["costBasis"] - 1) * 100 if port["costBasis"] else None
                positions.append({
                    "ticker": ticker,
                    "variant": variant,
                    "entryDate": port["entryDate"],
                    "entryPrice": port["entryPrice"],
                    "qty": port["qty"],
                    "lastClose": today_close,
                    "unrealizedPct": unreal,
                    "equityMTM": mtm,
                    "pendingExitSignalDate": port.get("pendingExitSignalDate"),
                })

        if n % 25 == 0:
            print(f"{n}/{len(TICKERS)}")

    state["updatedAtWITA"] = now.strftime("%Y-%m-%d %H:%M WITA")
    save_state(state)

    payload = {
        "updatedAtWITA": now.strftime("%Y-%m-%d %H:%M WITA"),
        "marketDate": market_date_global,
        "status": "UPDATED",
        "newBuy": new_buy,
        "newExit": new_exit,
        "pendingExecutions": pending_exec,
        "positions": positions,
        "tickerSnapshots": snapshots,
        "forwardState": state["portfolios"],
    }
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_FILE)

    print("DONE")
    print("Market date:", market_date_global)
    print("NEW BUY:", len(new_buy))
    print("NEW EXIT:", len(new_exit))
    print("Executed:", len(pending_exec))
    print("Open positions:", len(positions))

if __name__ == "__main__":
    main()
