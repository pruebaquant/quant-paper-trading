"""
record_slippage.py — Corre a las 10:00 ET (30 min post apertura)

Para cada trade abierto que fue registrado ayer (entry_date = ayer):
  1. Descarga el precio de apertura real de hoy
  2. Calcula el slippage vs precio de cierre de la señal
  3. Actualiza entry_price con el open real
  4. Registra en slippage_log.json

Slippage = (open_real - close_señal) / close_señal × 100
Positivo = pagaste más de lo esperado (malo)
Negativo = pagaste menos (bueno, raro)
"""

import os, json
from datetime import datetime, timedelta
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRADES_JSON = os.path.join(SCRIPT_DIR, 'trades.json')
DOCS_TRADES = os.path.join(SCRIPT_DIR, 'docs', 'trades.json')
SLIP_LOG    = os.path.join(SCRIPT_DIR, 'slippage_log.json')

def load_trades():
    if not os.path.exists(TRADES_JSON):
        return []
    with open(TRADES_JSON) as f:
        return json.load(f)

def save_trades(trades):
    for path in [TRADES_JSON, DOCS_TRADES]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(trades, f, indent=2)

def main():
    import yfinance as yf

    now   = datetime.utcnow()
    today = now.strftime('%Y-%m-%d')
    # Entry real = hoy (workflow corre morning del día hábil post-señal)
    # Buscamos trades donde entry_real = today o entry_date = ayer
    yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Recording open prices — {today} 10:00 ET")

    trades = load_trades()

    # Trades que necesitan el open de hoy
    # (registrados ayer, entry_real = today, sin open_price todavía)
    to_record = [
        t for t in trades
        if t['status'] == 'open'
        and (t.get('entry_real') == today or t.get('entry_date') == yesterday)
        and not t.get('open_price')
    ]

    if not to_record:
        print("  Sin trades para registrar open hoy.")
        return

    tickers = list({t['ticker'] for t in to_record})
    print(f"  Descargando opens para: {tickers}")

    try:
        raw = yf.download(tickers,
                          start=today, end=today,
                          auto_adjust=True, progress=False)
        if raw.empty:
            # Intentar con período más amplio
            raw = yf.download(tickers,
                              period='2d',
                              auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  ERROR descargando datos: {e}")
        return

    # Extraer opens
    opens = {}
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            op = raw['Open']
            for tkr in tickers:
                if tkr in op.columns:
                    s = op[tkr].dropna()
                    if not s.empty:
                        opens[tkr] = float(s.iloc[-1])
        else:
            if 'Open' in raw.columns:
                s = raw['Open'].dropna()
                if not s.empty and tickers:
                    opens[tickers[0]] = float(s.iloc[-1])
    except Exception as e:
        print(f"  ERROR procesando opens: {e}")

    print(f"  Opens obtenidos: {opens}")

    # Data quality check
    slip_records = []
    for t in to_record:
        tkr = t['ticker']
        open_price = opens.get(tkr)
        if not open_price:
            print(f"  [WARN] Sin open para {tkr}")
            continue

        # Slippage vs precio de cierre de la señal (entry_price original)
        close_signal = t.get('entry_price')
        slippage_pct = None
        slippage_usd = None

        if close_signal and close_signal > 0:
            slippage_pct = (open_price - close_signal) / close_signal * 100
            pos_usd      = t.get('pos_usd', 5000)
            shares_approx= pos_usd / close_signal
            slippage_usd = slippage_pct / 100 * pos_usd

            print(f"  {tkr}: close={close_signal:.2f} open={open_price:.2f} "
                  f"slippage={slippage_pct:+.3f}% (${slippage_usd:+.2f})")
        else:
            print(f"  {tkr}: sin precio de cierre base, registrando open={open_price:.2f}")

        # Actualizar trade con open real
        t['open_price']    = round(open_price, 4)
        t['slippage_pct']  = round(slippage_pct, 4) if slippage_pct else None
        t['slippage_usd']  = round(slippage_usd, 2) if slippage_usd else None

        # Usar open como precio de entrada real para el cálculo de retorno
        if open_price:
            t['entry_price_real'] = round(open_price, 4)

        slip_records.append({
            'date':          today,
            'ticker':        tkr,
            'close_signal':  close_signal,
            'open_real':     round(open_price, 4),
            'slippage_pct':  round(slippage_pct, 4) if slippage_pct else None,
            'slippage_usd':  round(slippage_usd, 2) if slippage_usd else None,
            'pos_usd':       t.get('pos_usd'),
        })

    # Guardar trades actualizados
    save_trades(trades)

    # Log de slippage histórico
    log = []
    if os.path.exists(SLIP_LOG):
        try:
            with open(SLIP_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.extend(slip_records)
    log = log[-365:]  # último año
    with open(SLIP_LOG, 'w') as f:
        json.dump(log, f, indent=2)

    # Summary
    if slip_records:
        avg_slip = sum(r['slippage_pct'] for r in slip_records if r['slippage_pct']) / len(slip_records)
        total_cost = sum(r['slippage_usd'] for r in slip_records if r['slippage_usd'])
        print(f"\n  Slippage promedio hoy: {avg_slip:+.3f}%")
        print(f"  Costo total slippage: ${total_cost:+.2f}")

if __name__ == '__main__':
    main()
