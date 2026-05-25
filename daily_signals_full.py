"""
daily_signals.py — Pipeline completo automatizado

Cada día a las 16:00 ET:
  1. Descarga precios de cierre
  2. Cierra trades vencidos con precio real
  3. Calcula P&L del día
  4. Genera señales nuevas y las registra
  5. Manda email con resumen
  6. Actualiza trades.json y signals_hoy.json
"""

from __future__ import annotations
import os, sys, json, pickle, warnings, time, smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(SCRIPT_DIR, 'model')
TRADES_JSON  = os.path.join(SCRIPT_DIR, 'trades.json')
OUT_JSON     = os.path.join(SCRIPT_DIR, 'signals_hoy.json')
LOG_JSON     = os.path.join(SCRIPT_DIR, 'signals_log.json')
DOCS_DIR     = os.path.join(SCRIPT_DIR, 'docs')
EARNINGS_PKL = os.path.join(MODEL_DIR, 'earnings_cache.pkl')

TC      = 0.0005  # 5bps one-way
TC_FIXED= 10      # $10 costo fijo por trade
MIN_POS = 3000    # minimo USD por posicion
HP_DAYS = 5       # holding period en dias habiles
CAPITAL = 50000   # capital nocional

# ── Universe filter v2 ────────────────────────────────────────────────────
FX_ETFS = {'FXE','FXB','FXF','FXY','FXC','FXA','UUP','UDN','CYB','CEW','DBV'}
FIXED_INCOME = {
    'AGG','BND','BNDX','TLT','IEF','SHY','GOVT','BSV','BIV','BLV',
    'LQD','HYG','JNK','USHY','TIP','MBB','MUB','EMB','MINT','JPST',
    'VCIT','VCLT','IGSB','IGIB','SPSB','SPIB','SPLB','VGIT','VGLT','VGSH',
}
COMMODITY = {'GLD','SLV','IAU','PDBC','DBC','GSG','USO','UNG','CORN','WEAT','CPER','OIH'}
LOW_LIQ   = {'DOYU','GRAB','GOTO','NGE'}
EXEC_EXCL = FX_ETFS | FIXED_INCOME | COMMODITY | LOW_LIQ

# Feriados NYSE 2025-2027
NYSE_HOL = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-04-26','2027-05-31',
}

def addBizDay(from_date: str, n: int = 1) -> str:
    """Siguiente día hábil respetando feriados NYSE."""
    from datetime import datetime as dt, timedelta as td
    d = dt.strptime(from_date, '%Y-%m-%d')
    added = 0
    while added < n:
        d += td(days=1)
        if d.weekday() < 5 and d.strftime('%Y-%m-%d') not in NYSE_HOL:
            added += 1
    return d.strftime('%Y-%m-%d')

FALLBACK_TICKERS = [
    'SPY','QQQ','IWM','EFA','EEM','IEFA','IEMG','EWK','EWG','EWU','EWJ',
    'IEV','FXI','INDA','EWZ','IJH','GOOG','GOOGL','INTC','AMD','AAPL',
    'MSFT','AMZN','META','NVDA','JPM','BAC','GS','ECL','HD','CNI','ABT',
    'BR','GIS','CAG','CPB','CNQ','COP','DHR','GILD','BMY','ABBV','COST',
    'MRK','PFE','LLY','CRM','ORCL','ADBE','XOM','CVX','EOG',
]

def _cls(t):
    t = t.upper()
    if t in FX_ETFS:       return 'FX_ETF'
    if t in FIXED_INCOME:  return 'FixedIncome_ETF'
    if t in COMMODITY:     return 'Commodity_ETF'
    if t in LOW_LIQ:       return 'LowLiquidity'
    if t in {'EEM','VWO','IEMG','EWZ','EWT','EWY','INDA','MCHI','FXI',
             'GXC','ASHR','EWH','EWS','ECH','EWW','EWM'}:
        return 'EM_ETF'
    if t in {'EFA','VEA','IEFA','EWJ','EWG','EWU','EWC','EWA','EWI',
             'EWQ','EWP','EWN','EWD','EWL','EWK','EWO','IEV','VGK','FEZ'}:
        return 'IntlDev_ETF'
    if t in {'SPY','QQQ','IWM','DIA','VTI','VOO','IVV','IJH','IJR',
             'XLK','XLF','XLV','XLI','XLY','XLP','XLE','XLU','XLB','XLRE','XLC'}:
        return 'USA_Equity_ETF'
    if len(t) <= 4 and not t.startswith('EW'):
        return 'USA_Equity'
    return 'Other_ETF'

# ── Trades JSON ───────────────────────────────────────────────────────────
def load_trades() -> list:
    if not os.path.exists(TRADES_JSON):
        return []
    with open(TRADES_JSON) as f:
        return json.load(f)

def save_trades(trades: list):
    with open(TRADES_JSON, 'w') as f:
        json.dump(trades, f, indent=2)
    # Copiar a docs/ para GitHub Pages
    docs_trades = os.path.join(DOCS_DIR, 'trades.json')
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(docs_trades, 'w') as f:
        json.dump(trades, f, indent=2)

# ── Fetch closing prices ──────────────────────────────────────────────────
def fetch_close(tickers: list, date_str: str = None) -> dict:
    """Descarga precio de cierre para una lista de tickers."""
    import yfinance as yf
    if not tickers:
        return {}
    end   = datetime.today()
    start = (end - timedelta(days=10)).strftime('%Y-%m-%d')
    prices = {}
    try:
        raw = yf.download(tickers, start=start,
                          end=end.strftime('%Y-%m-%d'),
                          auto_adjust=True, progress=False)
        if raw.empty:
            return {}
        close = raw['Close'] if 'Close' in raw.columns else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        for tkr in tickers:
            try:
                s = close[tkr].dropna()
                if not s.empty:
                    prices[tkr] = float(s.iloc[-1])
            except Exception:
                pass
    except Exception as e:
        print(f"  [WARN] fetch_close error: {e}")
    return prices

# ── Close expired trades ──────────────────────────────────────────────────
def close_expired_trades(trades: list, today: str) -> tuple[list, list]:
    """
    Cierra trades cuya fecha de salida esperada <= hoy.
    Retorna (trades_actualizados, trades_cerrados_hoy).
    """
    today_dt = datetime.strptime(today, '%Y-%m-%d').date()
    to_close = [t for t in trades
                if t['status'] == 'open'
                and t.get('exit_expected')
                and datetime.strptime(t['exit_expected'], '%Y-%m-%d').date() <= today_dt]

    if not to_close:
        return trades, []

    # Descargar precios de cierre para los tickers a cerrar
    tickers_to_close = list({t['ticker'] for t in to_close})
    print(f"  Cerrando {len(to_close)} trades vencidos: {tickers_to_close}")
    close_prices = fetch_close(tickers_to_close)

    closed_today = []
    for trade in trades:
        if trade not in to_close:
            continue
        tkr         = trade['ticker']
        entry_price = trade.get('entry_price')
        exit_price  = close_prices.get(tkr)

        if exit_price and entry_price:
            ret_gross = (exit_price / entry_price - 1) * 100
        elif exit_price is None:
            ret_gross = 0.0  # sin precio disponible
            exit_price = None
        else:
            ret_gross = 0.0

        ret_net = ret_gross - TC * 100

        trade['status']     = 'win' if ret_net > 0 else 'loss'
        trade['exit_date']  = today
        trade['exit_price'] = round(exit_price, 2) if exit_price else None
        trade['ret']        = round(ret_gross, 3)
        trade['ret_net']    = round(ret_net, 3)
        trade['closed_by']  = 'auto'
        closed_today.append(trade)

    return trades, closed_today

# ── Model loading ─────────────────────────────────────────────────────────
def load_model():
    import xgboost as xgb
    required = ['buy_model.ubj', 'buy_thresholds.json',
                'feature_names.json', 'cfg.json']
    for fn in required:
        if not os.path.exists(os.path.join(MODEL_DIR, fn)):
            print(f"ERROR: model/{fn} no encontrado.")
            sys.exit(1)
    buy_model = xgb.Booster()
    buy_model.load_model(os.path.join(MODEL_DIR, 'buy_model.ubj'))
    with open(os.path.join(MODEL_DIR, 'buy_thresholds.json')) as f:
        buy_thr = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.json')) as f:
        feat_names = json.load(f)
    with open(os.path.join(MODEL_DIR, 'cfg.json')) as f:
        cfg = json.load(f)
    meta = {}
    if os.path.exists(os.path.join(MODEL_DIR, 'meta.json')):
        with open(os.path.join(MODEL_DIR, 'meta.json')) as f:
            meta = json.load(f)
    print(f"  Modelo: entrenado {meta.get('trained_on','?')[:10]}")
    return {'buy_model': buy_model, 'buy_thr': buy_thr,
            'feat_names': feat_names, 'cfg': cfg}

# ── Download + features ───────────────────────────────────────────────────
def download_data(tickers, days=420):
    import yfinance as yf
    end   = datetime.today()
    start = (end - timedelta(days=days)).strftime('%Y-%m-%d')
    print(f"  Descargando {len(tickers)} tickers...")
    frames = []
    BATCH  = 25
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            raw = yf.download(batch, start=start, end=end.strftime('%Y-%m-%d'),
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for tkr in batch:
                    try:
                        df_t = raw.xs(tkr, axis=1, level=1).copy()
                        df_t.columns = [c.lower() for c in df_t.columns]
                        df_t.index.name = 'date'
                        df_t['ticker'] = tkr
                        frames.append(df_t.reset_index().set_index(['ticker','date']))
                    except KeyError:
                        pass
            else:
                df_t = raw.copy()
                df_t.columns = [c.lower() for c in df_t.columns]
                df_t.index.name = 'date'
                df_t['ticker'] = batch[0]
                frames.append(df_t.reset_index().set_index(['ticker','date']))
        except Exception as e:
            print(f"    Batch {i//BATCH+1} error: {e}")
        time.sleep(0.3)
    if not frames:
        print("ERROR: sin datos")
        sys.exit(1)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    print(f"  OK: {len(df):,} filas | {df.index.get_level_values('ticker').nunique()} tickers")
    return df

def compute_features(df):
    out = {}
    for tkr in df.index.get_level_values('ticker').unique():
        try:
            t = df.xs(tkr, level='ticker').copy().sort_index()
            if len(t) < 60: continue
            c,h,l,o,v = t['close'],t['high'],t['low'],t['open'],t['volume']
            ret = c.pct_change()
            u  = np.log(h/o.clip(lower=1e-9))
            d  = np.log(l/o.clip(lower=1e-9))
            c2 = np.log(c/o.clip(lower=1e-9))
            gk = 0.5*(u-d)**2-(2*np.log(2)-1)*c2**2
            iv = gk.rolling(252,min_periods=126).mean().apply(np.sqrt)
            v5 = gk.rolling(5,min_periods=3).mean().apply(np.sqrt)
            vov= iv.rolling(252,min_periods=126).std()
            vr = v5/iv.clip(lower=1e-9)
            m12= c.pct_change(252)/c.pct_change(21).clip(-1,None)
            mz = (m12-m12.rolling(252,min_periods=126).mean()) / \
                  m12.rolling(252,min_periods=126).std().clip(lower=1e-9)
            feat = pd.DataFrame({
                'mom_12_1':m12,'tsmom_12':c.pct_change(252).clip(-1,1),
                'ret_1m':c.pct_change(21),'ret_5d':c.pct_change(5),
                'ret_10d':c.pct_change(10),
                'high_52w':c/c.rolling(252,min_periods=126).max(),
                'mom_12_1_zscore':mz,'ivol_12m':iv,'vol_of_vol':vov,
                'vol_5d':v5,'vol_ratio':vr,
                'rolling_beta':ret.rolling(252,min_periods=126).corr(ret.shift(1)).clip(-3,3),
                'amihud':(ret.abs()/v.replace(0,np.nan)*1e6).rolling(21,min_periods=10).mean(),
                'log_volume':np.log1p(v),
            }, index=t.index)
            feat['ticker'] = tkr
            out[tkr] = feat.reset_index().set_index(['ticker','date'])
        except Exception:
            pass
    if not out: return pd.DataFrame()
    df_f = pd.concat(out.values()).sort_index()
    for base,cs in [('ivol_12m','ivol_12m_cs'),('mom_12_1','mom_12_1_cs'),
                    ('rolling_beta','rolling_beta_cs'),('vol_of_vol','vol_of_vol_cs'),
                    ('ret_1m','ret_1m_cs')]:
        if base in df_f.columns:
            grp = df_f.groupby(level='date')[base]
            df_f[cs]=(df_f[base]-grp.transform('mean'))/grp.transform('std').clip(lower=1e-9)
    df_f['mom_x_volratio']    = df_f['mom_12_1']*df_f['vol_ratio']
    df_f['tsmom_x_vov']       = df_f['tsmom_12']/df_f['vol_of_vol'].clip(lower=1e-9)
    df_f['ret1m_x_highvol']   = df_f['ret_1m']*(df_f['vol_ratio']>1).astype(float)
    df_f['mom_cs_x_volratio'] = df_f['mom_12_1_cs']*df_f['vol_ratio']
    return df_f

def compute_regime(df_f):
    q = df_f['vol_ratio'].groupby(level='date').transform(lambda x: x.rank(pct=True))
    return q.map(lambda x: 'low_vol' if x<0.33 else ('high_vol' if x>0.67 else 'neutral'))

def compute_zscore(df_f):
    disp = df_f['ret_1m'].groupby(level='date').std()
    mu   = disp.rolling(252,min_periods=126).mean()
    sd   = disp.rolling(252,min_periods=126).std().clip(lower=1e-9)
    return (disp-mu)/sd

def load_blackout():
    if not os.path.exists(EARNINGS_PKL): return {}
    with open(EARNINGS_PKL,'rb') as f:
        cache = pickle.load(f)
    blackout = {}
    for tkr,dates in cache.items():
        blocked = set()
        for d in dates:
            try:
                dn = pd.Timestamp(d).tz_localize(None).normalize() \
                     if (hasattr(d,'tz') and d.tz) else pd.Timestamp(d).normalize()
                for off in range(-2,3):
                    blocked.add(dn+pd.Timedelta(days=off))
            except Exception:
                pass
        blackout[tkr] = blocked
    return blackout

# ── Email ─────────────────────────────────────────────────────────────────
def send_email(subject: str, html_body: str):
    """Envía email via Gmail SMTP. Requiere secrets GMAIL_USER y GMAIL_PASS."""
    sender   = os.environ.get('GMAIL_USER')
    password = os.environ.get('GMAIL_PASS')
    receiver = os.environ.get('NOTIFY_EMAIL', sender)

    if not sender or not password:
        print("  [INFO] Email no configurado (GMAIL_USER/GMAIL_PASS no definidos)")
        return

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = sender
        msg['To']      = receiver
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, receiver, msg.as_string())
        print(f"  Email enviado a {receiver}")
    except Exception as e:
        print(f"  [WARN] Email error: {e}")

def build_email(date, zscore, closed_today, new_signals, all_trades, system_active, sector_alert_html='', dq_warning_html=''):
    """Construye el HTML del email diario."""
    # Stats generales
    all_closed = [t for t in all_trades if t['status'] != 'open']
    all_open   = [t for t in all_trades if t['status'] == 'open']
    wins       = [t for t in all_closed if t['status'] == 'win']
    wr         = len(wins)/len(all_closed)*100 if all_closed else 0
    pnl        = sum(t['ret_net'] for t in all_closed if t.get('ret_net') is not None)
    avg        = pnl/len(all_closed) if all_closed else 0

    z_color  = '#22c55e' if zscore and zscore > 0 else '#ef4444'
    z_str    = f"{zscore:+.3f}" if zscore else '—'
    z_status = 'ACTIVO' if zscore and zscore > 0 else 'INACTIVO'

    # Trades cerrados hoy
    closed_html = ''
    if closed_today:
        rows = ''
        for t in closed_today:
            ret    = t.get('ret_net', 0)
            color  = '#22c55e' if ret > 0 else '#ef4444'
            result = 'WIN ✓' if ret > 0 else 'LOSS ✗'
            rows += f"""
            <tr>
              <td style="padding:8px 12px;font-weight:600">{t['ticker']}</td>
              <td style="padding:8px 12px;color:#888">{t['entry_date']}</td>
              <td style="padding:8px 12px;color:#888">{t['exit_date']}</td>
              <td style="padding:8px 12px;color:{color};font-weight:600">{ret:+.2f}%</td>
              <td style="padding:8px 12px;color:{color}">{result}</td>
            </tr>"""
        closed_html = f"""
        <h3 style="color:#c8cdd6;font-size:14px;margin:24px 0 12px;font-family:monospace">
          TRADES CERRADOS HOY ({len(closed_today)})
        </h3>
        <table style="width:100%;border-collapse:collapse;background:#111318;border-radius:6px;overflow:hidden">
          <thead>
            <tr style="background:#1e2229">
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">TICKER</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">ENTRADA</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">SALIDA</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">RET NETO</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">RESULTADO</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""

    # Señales nuevas
    signals_html = ''
    if new_signals and system_active:
        rows = ''
        for s in new_signals:
            rows += f"""
            <tr>
              <td style="padding:8px 12px;font-weight:600">{s['ticker']}</td>
              <td style="padding:8px 12px;color:#3b82f6">{s['p_buy']:.3f}</td>
              <td style="padding:8px 12px;color:#888">{s['asset_type']}</td>
              <td style="padding:8px 12px;color:#888">${s['entry_price'] or '—'}</td>
              <td style="padding:8px 12px;color:#888">{s['exit_expected']}</td>
            </tr>"""
        signals_html = f"""
        <h3 style="color:#c8cdd6;font-size:14px;margin:24px 0 12px;font-family:monospace">
          SEÑALES NUEVAS ({len(new_signals)}) — registradas automáticamente
        </h3>
        <table style="width:100%;border-collapse:collapse;background:#111318;border-radius:6px;overflow:hidden">
          <thead>
            <tr style="background:#1e2229">
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">TICKER</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">p_buy</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">TIPO</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">PRECIO</th>
              <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">SAL. ESP.</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""
    elif not system_active:
        signals_html = f"""
        <div style="padding:14px 18px;background:#5c3a0a;border:1px solid #f59e0b;
                    border-radius:6px;margin-top:16px;font-family:monospace;font-size:13px">
          ⚠ Sistema inactivo hoy — zscore={z_str} (necesita &gt; 0). Sin señales nuevas.
        </div>"""

    # Posiciones abiertas
    open_html = ''
    if all_open:
        rows = ''
        for t in all_open:
            rows += f"""
            <tr>
              <td style="padding:6px 12px;font-weight:600">{t['ticker']}</td>
              <td style="padding:6px 12px;color:#888">{t['entry_date']}</td>
              <td style="padding:6px 12px;color:#f59e0b">{t['exit_expected']}</td>
              <td style="padding:6px 12px;color:#3b82f6">{t.get('pbuy') or t.get('p_buy','—')}</td>
            </tr>"""
        open_html = f"""
        <h3 style="color:#c8cdd6;font-size:14px;margin:24px 0 12px;font-family:monospace">
          POSICIONES ABIERTAS ({len(all_open)})
        </h3>
        <table style="width:100%;border-collapse:collapse;background:#111318;border-radius:6px;overflow:hidden">
          <thead><tr style="background:#1e2229">
            <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">TICKER</th>
            <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">ENTRADA</th>
            <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">SAL. ESP.</th>
            <th style="padding:8px 12px;text-align:left;color:#545c6b;font-size:11px;font-weight:400">p_buy</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    pnl_color = '#22c55e' if pnl >= 0 else '#ef4444'

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="background:#0a0c0f;color:#c8cdd6;font-family:Arial,sans-serif;
             margin:0;padding:0">
  <div style="max-width:600px;margin:0 auto;padding:32px 24px">

    <!-- Header -->
    <div style="border-bottom:1px solid #2a2f38;padding-bottom:20px;margin-bottom:24px">
      <h1 style="font-family:monospace;font-size:20px;font-weight:500;
                 color:#6ee7b7;margin:0">▸ PAPER TRADING — {date}</h1>
      <p style="font-size:11px;color:#545c6b;margin:6px 0 0;
                letter-spacing:.08em;text-transform:uppercase">
        HP=5d · Universe Filter v2 · Reporte automático diario
      </p>
    </div>

    <!-- Zscore banner -->
    <div style="padding:14px 18px;background:{'#16532d' if zscore and zscore>0 else '#5c1616'};
                border:1px solid {z_color};border-radius:6px;
                margin-bottom:24px;font-family:monospace">
      <span style="color:{z_color};font-weight:600">{z_status}</span>
      <span style="color:#888;margin-left:12px">zscore dispersión = {z_str}</span>
    </div>

    <!-- Stats -->
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                gap:1px;background:#1e2229;border-radius:6px;
                overflow:hidden;margin-bottom:24px">
      <div style="background:#111318;padding:14px 16px">
        <div style="font-size:10px;color:#545c6b;text-transform:uppercase;letter-spacing:.1em">Trades totales</div>
        <div style="font-size:22px;font-weight:600;color:#fff;margin-top:4px">{len(all_trades)}</div>
        <div style="font-size:10px;color:#363d4a">{len(all_open)} abiertos</div>
      </div>
      <div style="background:#111318;padding:14px 16px">
        <div style="font-size:10px;color:#545c6b;text-transform:uppercase;letter-spacing:.1em">Win Rate</div>
        <div style="font-size:22px;font-weight:600;
                    color:{'#22c55e' if wr>=50 else '#ef4444'};margin-top:4px">
          {wr:.1f}%
        </div>
        <div style="font-size:10px;color:#363d4a">{len(wins)}G / {len(all_closed)-len(wins)}P</div>
      </div>
      <div style="background:#111318;padding:14px 16px">
        <div style="font-size:10px;color:#545c6b;text-transform:uppercase;letter-spacing:.1em">Avg Ret/Trade</div>
        <div style="font-size:22px;font-weight:600;
                    color:{'#22c55e' if avg>=0 else '#ef4444'};margin-top:4px">
          {avg:+.3f}%
        </div>
        <div style="font-size:10px;color:#363d4a">neto 5bps</div>
      </div>
      <div style="background:#111318;padding:14px 16px">
        <div style="font-size:10px;color:#545c6b;text-transform:uppercase;letter-spacing:.1em">PnL acum.</div>
        <div style="font-size:22px;font-weight:600;color:{pnl_color};margin-top:4px">
          {pnl:+.2f}%
        </div>
        <div style="font-size:10px;color:#363d4a">suma retornos</div>
      </div>
    </div>

    {closed_html}
    {signals_html}
    {open_html}

    <!-- Sector alerts -->
    {sector_alert_html}

    <!-- DQ warning -->
    {dq_warning_html}

    <!-- OOS contamination disclaimer -->
    <div style="margin-top:24px;padding:10px 14px;background:#1c2535;border-radius:4px;
                font-size:10px;color:#545c6b;font-family:monospace">
      ⚠ NOTA METODOLÓGICA: El OOS 2023-2026 fue parcialmente observado durante el desarrollo.
      El único test verdaderamente limpio es el paper trading en curso.
      Retornos nominales en USD. Costo de oportunidad del cash no incluido en P&L reportado.
    </div>

    <!-- Footer -->
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #1e2229;
                font-size:11px;color:#363d4a;font-family:monospace">
      Dashboard: https://pruebaquant.github.io/quant-paper-trading/<br>
      Generado automáticamente por GitHub Actions · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
    </div>
  </div>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    now   = datetime.utcnow()
    today = now.strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"DAILY SIGNALS — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print('='*60)

    # 1. Cargar trades existentes
    print("\n1. Cargando trades...")
    trades = load_trades()
    open_trades = [t for t in trades if t['status'] == 'open']
    print(f"   {len(trades)} trades totales | {len(open_trades)} abiertos")

    # 2. Cerrar trades vencidos
    print("\n2. Cerrando trades vencidos...")
    trades, closed_today = close_expired_trades(trades, today)
    if closed_today:
        for t in closed_today:
            ret = t.get('ret_net', 0)
            mark = '✓ WIN' if ret > 0 else '✗ LOSS'
            print(f"   {mark}  {t['ticker']:<8} {ret:+.2f}%  "
                  f"(entrada={t.get('entry_price','?')} salida={t.get('exit_price','?')})")
    else:
        print("   Sin trades vencidos hoy.")

    # 3. Cargar modelo y datos
    print("\n3. Cargando modelo...")
    md = load_model()

    tickers = sorted({t.upper() for t in FALLBACK_TICKERS
                      if t.upper() not in EXEC_EXCL})

    print("\n4. Descargando datos...")
    df_raw = download_data(tickers)

    print("\n5. Data quality check...")
    dq_issues = []
    for tkr in df_raw.index.get_level_values('ticker').unique():
        try:
            t = df_raw.xs(tkr, level='ticker')
            # Check 1: precio razonable (no gaps > 30% en un dia)
            ret = t['close'].pct_change().abs()
            if (ret > 0.30).any():
                dq_issues.append(f"{tkr}: gap >30%% detectado")
            # Check 2: sin volumen
            if (t['volume'] == 0).sum() > 5:
                dq_issues.append(f"{tkr}: >5 dias sin volumen")
            # Check 3: precio negativo o cero
            if (t['close'] <= 0).any():
                dq_issues.append(f"{tkr}: precio <= 0")
        except Exception:
            pass

    if dq_issues:
        print(f"   ⚠ Data quality issues ({len(dq_issues)}):")
        for issue in dq_issues[:10]:
            print(f"     {issue}")
    else:
        print(f"   ✓ Data quality OK — sin anomalías detectadas")

    print("\n6. Calculando features y régimen...")
    df_f    = compute_features(df_raw)
    regime  = compute_regime(df_f)
    zscore  = compute_zscore(df_f)
    latest  = df_f.index.get_level_values('date').max()
    z_today = float(zscore.loc[latest]) if latest in zscore.index else float('nan')

    print(f"   Fecha: {latest.date()}  |  Zscore: {z_today:+.3f}")

    entry_date = latest.strftime('%Y-%m-%d')
    exit_date  = (latest + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    new_signals = []
    system_active = False

    if z_today == z_today and z_today > 0:
        system_active = True
        print("\n6. Generando señales...")

        import xgboost as xgb
        today_mask = df_f.index.get_level_values('date') == latest
        df_today   = df_f[today_mask].copy()
        feat_names = md['feat_names']
        buy_thr    = md['buy_thr']
        cfg        = md['cfg']

        for fn in feat_names:
            if fn not in df_today.columns:
                df_today[fn] = 0.0

        X       = df_today[feat_names].fillna(0).values.astype(float)
        dm      = xgb.DMatrix(X, feature_names=feat_names)
        p_arr   = md['buy_model'].predict(dm)
        p_buy   = pd.Series(p_arr, index=df_today.index)
        reg_tod = regime[today_mask]

        def get_thr(reg):
            return float(buy_thr.get(str(reg), buy_thr.get('global', 0.50)))

        thr_s = reg_tod.map(get_thr)
        sigs  = pd.DataFrame({'p_buy':p_buy,'regime':reg_tod}, index=df_today.index)
        sigs  = sigs[sigs['p_buy'] >= thr_s]
        if cfg.get('block_neutral', True):
            sigs = sigs[sigs['regime'] != 'neutral']
        if cfg.get('block_low_vol', True):
            sigs = sigs[sigs['regime'] != 'low_vol']

        tkrs = sigs.index.get_level_values('ticker')
        sigs = sigs[~tkrs.isin(EXEC_EXCL)]

        # Earnings blackout
        blackout = load_blackout()
        today_ts = pd.Timestamp(latest).normalize()
        to_drop  = [(t,d) for t,d in sigs.index
                    if any((today_ts+pd.Timedelta(days=o)) in blackout.get(t,set())
                           for o in range(6))]
        sigs = sigs.drop(to_drop, errors='ignore')

        # Top N — sin contar posiciones ya abiertas en mismo ticker
        open_tickers = {t['ticker'] for t in trades if t['status'] == 'open'}
        sigs = sigs[~sigs.index.get_level_values('ticker').isin(open_tickers)]

        top_n = int(cfg.get('top_n_per_date', 5))
        open_count = len([t for t in trades if t['status'] == 'open'])
        slots = max(0, top_n - open_count)

        if slots > 0 and len(sigs) > slots:
            sigs = sigs.nlargest(slots, 'p_buy')
        elif slots == 0:
            print(f"   Ya hay {open_count} posiciones abiertas (top_n={top_n}). Sin slots.")
            sigs = sigs.iloc[0:0]

        # Registrar señales como trades abiertos
        for (tkr, _), row in sigs.iterrows():
            try:
                price = float(df_raw.xs(tkr,level='ticker')['close'].iloc[-1])
            except Exception:
                price = None

            entry_real = addBizDay(entry_date)
            # Top features para audit trail
            audit_feats = {}
            for feat_k in ['vol_ratio','mom_12_1_cs','ret_1m','vol_5d','ivol_12m']:
                try:
                    audit_feats[feat_k] = round(float(df_today.loc[(tkr,latest), feat_k]), 4)
                except Exception:
                    audit_feats[feat_k] = None

            signal = {
                'ticker':        tkr,
                'entry_date':    entry_date,
                'entry_real':    entry_real,
                'exit_expected': addBizDay(entry_real, HP_DAYS),
                'p_buy':         round(float(row['p_buy']), 4),
                'pbuy':          round(float(row['p_buy']), 4),
                'regime':        str(row['regime']),
                'asset_type':    _cls(tkr),
                'entry_price':   round(price, 2) if price else None,
                'zscore':        round(z_today, 3),
                'status':        'open',
                'exit_date':     None,
                'exit_price':    None,
                'ret':           None,
                'ret_net':       None,
                'pnl_usd':       None,
                'tc_usd':        None,
                'registered_by': 'auto',
                'id':            int(datetime.utcnow().timestamp() * 1000) + len(new_signals),
                'audit': {
                    'features':     audit_feats,
                    'dq_issues':    len(dq_issues),
                    'generated_at': now.isoformat() + 'Z',
                }
            }
            trades.append(signal)
            new_signals.append(signal)
            print(f"   ✓ {tkr:<8} p_buy={row['p_buy']:.3f}  {row['regime']}")

        if not new_signals:
            print("   Sin señales nuevas tras filtros.")
    else:
        print(f"\n   ⚠ Sistema inactivo (zscore={z_today:.3f})")

    # 7. Guardar trades y signals
    print("\n7. Guardando archivos...")
    save_trades(trades)

    # SPY return del dia para benchmark
    spy_ret_today = None
    try:
        import yfinance as yf
        spy = yf.download('SPY', period='2d', auto_adjust=True, progress=False)
        if not spy.empty and len(spy) >= 2:
            spy_ret_today = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-2] - 1) * 100)
    except Exception:
        pass

    # Tasa libre de riesgo anualizada (aproximacion con T-bill 3m)
    risk_free_annual = 5.0  # % — actualizar manualmente si cambia significativamente

    payload = {
        'generated_at':  now.isoformat() + 'Z',
        'date':          entry_date,
        'zscore':        round(z_today, 3) if z_today == z_today else None,
        'system_active': system_active,
        'n_signals':     len(new_signals),
        'signals':       new_signals,
        'closed_today':  len(closed_today),
        'spy_ret_today': round(spy_ret_today, 4) if spy_ret_today else None,
        'risk_free_annual': risk_free_annual,
        'dq_issues':     len(dq_issues),
    }
    for path in [OUT_JSON, os.path.join(DOCS_DIR, 'signals_hoy.json')]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    # Log histórico
    log = []
    if os.path.exists(LOG_JSON):
        try:
            with open(LOG_JSON) as f: log = json.load(f)
        except Exception: pass
    log.append(payload)
    log = log[-90:]
    with open(LOG_JSON, 'w') as f:
        json.dump(log, f, indent=2)

    # 8. Email
    print("\n8. Enviando email...")
    subject = f"[Paper Trading] {entry_date} — "
    if closed_today:
        wins_hoy = sum(1 for t in closed_today if t.get('ret_net',0) > 0)
        subject += f"{wins_hoy}/{len(closed_today)} wins"
        if new_signals:
            subject += f" · {len(new_signals)} señales nuevas"
    elif new_signals:
        subject += f"{len(new_signals)} señales nuevas"
    elif not system_active:
        subject += "sistema inactivo"
    else:
        subject += "sin novedades"

    all_closed = [t for t in trades if t['status'] != 'open']
    # Sector concentration alert
    all_open = [t for t in trades if t['status'] == 'open']
    sectors = {}
    sector_map = {
        'NVDA':'Technology','AMD':'Technology','INTC':'Technology',
        'AAPL':'Technology','MSFT':'Technology','META':'Technology',
        'GOOG':'Technology','GOOGL':'Technology','AMZN':'Technology',
        'EEM':'EM','VWO':'EM','IEMG':'EM','EWZ':'EM','INDA':'EM','FXI':'EM',
        'EFA':'Intl Dev','IEFA':'Intl Dev','EWK':'Intl Dev','EWG':'Intl Dev',
        'EWU':'Intl Dev','EWJ':'Intl Dev','IEV':'Intl Dev',
    }
    for t in all_open:
        sec = sector_map.get(t['ticker'], 'Other')
        pos = t.get('pos_usd', 0) or 0
        sectors[sec] = sectors.get(sec, 0) + pos
    total_dep = sum(sectors.values())
    sector_alerts = [
        f"{sec}: {usd/total_dep*100:.0f}% (${usd:.0f})"
        for sec, usd in sectors.items()
        if total_dep > 0 and usd/CAPITAL > 0.40
    ]
    sector_alert_html = ''
    if sector_alerts:
        sector_alert_html = f'''<div style="margin-top:16px;padding:12px 16px;background:#2e0a15;border:1px solid #f43f5e;border-radius:6px;font-family:monospace;font-size:12px">
          <strong style="color:#f43f5e">⚠ CONCENTRACIÓN SECTORIAL</strong><br>
          <span style="color:#c9d1dd">{" | ".join(sector_alerts)}</span><br>
          <span style="color:#5a6a80;font-size:10px">Limite: 40% por sector. Revisar posiciones.</span>
        </div>'''

    # DQ warning
    dq_warning_html = ''
    if len(dq_issues) > 0:
        dq_warning_html = f'<div style="margin-top:12px;padding:10px 14px;background:#2e1f08;border:1px solid #f59e0b;border-radius:4px;font-size:11px;color:#f59e0b;font-family:monospace">⚠ Data quality: {len(dq_issues)} issue(s) detectados hoy. Señales pueden ser menos confiables.</div>'

    html = build_email(entry_date, z_today, closed_today, new_signals,
                       trades, system_active,
                       sector_alert_html=sector_alert_html,
                       dq_warning_html=dq_warning_html)
    send_email(subject, html)

    print(f"\n{'='*60}")
    print(f"  COMPLETADO: {len(closed_today)} trades cerrados | "
          f"{len(new_signals)} señales nuevas")
    print('='*60)

if __name__ == '__main__':
    main()
