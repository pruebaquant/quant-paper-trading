"""
daily_signals.py — Corre en GitHub Actions cada día a las 16:00 ET.

Usa modelo portable (XGBoost nativo .ubj) sin dependencias del pipeline local.
"""

from __future__ import annotations
import os, sys, json, pickle, warnings, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(SCRIPT_DIR, 'model')
OUT_JSON     = os.path.join(SCRIPT_DIR, 'signals_hoy.json')
LOG_JSON     = os.path.join(SCRIPT_DIR, 'signals_log.json')
DOCS_JSON    = os.path.join(SCRIPT_DIR, 'docs', 'signals_hoy.json')
EARNINGS_PKL = os.path.join(MODEL_DIR, 'earnings_cache.pkl')

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

FALLBACK_TICKERS = [
    'SPY','QQQ','IWM','EFA','EEM','IEFA','IEMG','EWK','EWG','EWU','EWJ',
    'IEV','FXI','INDA','EWZ','IJH','GOOG','GOOGL','INTC','AMD','AAPL',
    'MSFT','AMZN','META','NVDA','JPM','BAC','GS','ECL','HD','CNI','ABT',
    'BR','GIS','CAG','CPB','CNQ','COP','DHR','GILD','BMY','ABBV','COST',
    'MRK','PFE','LLY','CRM','ORCL','ADBE','XOM','CVX','EOG',
]

# ── Load portable model ────────────────────────────────────────────────────
def load_model():
    import xgboost as xgb

    required = ['buy_model.ubj','buy_thresholds.json',
                'feature_names.json','cfg.json']
    for fn in required:
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path):
            print(f"ERROR: {path} no encontrado.")
            print("Correr export_model.py y subir la carpeta model/ al repo.")
            sys.exit(1)

    # BUY model
    buy_model = xgb.Booster()
    buy_model.load_model(os.path.join(MODEL_DIR, 'buy_model.ubj'))

    # SELL model (opcional — para auditoría)
    sell_model = None
    sell_path  = os.path.join(MODEL_DIR, 'sell_model.ubj')
    if os.path.exists(sell_path):
        sell_model = xgb.Booster()
        sell_model.load_model(sell_path)

    with open(os.path.join(MODEL_DIR, 'buy_thresholds.json'))  as f:
        buy_thr = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.json'))   as f:
        feat_names = json.load(f)
    with open(os.path.join(MODEL_DIR, 'cfg.json'))             as f:
        cfg = json.load(f)

    meta_path = os.path.join(MODEL_DIR, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    print(f"  Modelo cargado (entrenado: {meta.get('trained_on','?')[:10]})")
    print(f"  Features: {len(feat_names)}  |  Thresholds: {buy_thr}")

    return {
        'buy_model':   buy_model,
        'sell_model':  sell_model,
        'buy_thr':     buy_thr,
        'feat_names':  feat_names,
        'cfg':         cfg,
    }

# ── Download data ──────────────────────────────────────────────────────────
def download_data(tickers, days=420):
    import yfinance as yf
    end   = datetime.today()
    start = (end - timedelta(days=days)).strftime('%Y-%m-%d')
    end_s = end.strftime('%Y-%m-%d')
    print(f"  Descargando {len(tickers)} tickers ({start} → {end_s})...")
    frames = []
    BATCH  = 25
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            raw = yf.download(batch, start=start, end=end_s,
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
        print("ERROR: sin datos descargados.")
        sys.exit(1)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    print(f"  OK: {len(df):,} filas | {df.index.get_level_values('ticker').nunique()} tickers")
    return df

# ── Features ───────────────────────────────────────────────────────────────
def compute_features(df):
    out = {}
    for tkr in df.index.get_level_values('ticker').unique():
        try:
            t = df.xs(tkr, level='ticker').copy().sort_index()
            if len(t) < 60:
                continue
            c,h,l,o,v = t['close'],t['high'],t['low'],t['open'],t['volume']
            ret = c.pct_change()
            u   = np.log(h/o.clip(lower=1e-9))
            d   = np.log(l/o.clip(lower=1e-9))
            c2  = np.log(c/o.clip(lower=1e-9))
            gk  = 0.5*(u-d)**2 - (2*np.log(2)-1)*c2**2
            iv  = gk.rolling(252,min_periods=126).mean().apply(np.sqrt)
            v5  = gk.rolling(5,  min_periods=3).mean().apply(np.sqrt)
            vov = iv.rolling(252,min_periods=126).std()
            vr  = v5/iv.clip(lower=1e-9)
            m12 = c.pct_change(252)/c.pct_change(21).clip(-1,None)
            mz  = (m12-m12.rolling(252,min_periods=126).mean()) / \
                   m12.rolling(252,min_periods=126).std().clip(lower=1e-9)
            feat = pd.DataFrame({
                'mom_12_1':        m12,
                'tsmom_12':        c.pct_change(252).clip(-1,1),
                'ret_1m':          c.pct_change(21),
                'ret_5d':          c.pct_change(5),
                'ret_10d':         c.pct_change(10),
                'high_52w':        c/c.rolling(252,min_periods=126).max(),
                'mom_12_1_zscore': mz,
                'ivol_12m':        iv,
                'vol_of_vol':      vov,
                'vol_5d':          v5,
                'vol_ratio':       vr,
                'rolling_beta':    ret.rolling(252,min_periods=126).corr(ret.shift(1)).clip(-3,3),
                'amihud':          (ret.abs()/v.replace(0,np.nan)*1e6).rolling(21,min_periods=10).mean(),
                'log_volume':      np.log1p(v),
            }, index=t.index)
            feat['ticker'] = tkr
            out[tkr] = feat.reset_index().set_index(['ticker','date'])
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    df_f = pd.concat(out.values()).sort_index()
    for base,cs in [('ivol_12m','ivol_12m_cs'),('mom_12_1','mom_12_1_cs'),
                    ('rolling_beta','rolling_beta_cs'),('vol_of_vol','vol_of_vol_cs'),
                    ('ret_1m','ret_1m_cs')]:
        if base in df_f.columns:
            grp = df_f.groupby(level='date')[base]
            df_f[cs] = (df_f[base]-grp.transform('mean'))/grp.transform('std').clip(lower=1e-9)
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
    if not os.path.exists(EARNINGS_PKL):
        return {}
    with open(EARNINGS_PKL,'rb') as f:
        cache = pickle.load(f)
    blackout = {}
    for tkr, dates in cache.items():
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

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    now = datetime.utcnow()
    print(f"\n{'='*60}")
    print(f"DAILY SIGNALS — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print('='*60)

    # 1. Modelo
    print("\n1. Cargando modelo portable...")
    md         = load_model()
    buy_model  = md['buy_model']
    buy_thr    = md['buy_thr']
    feat_names = md['feat_names']
    cfg        = md['cfg']

    tickers = sorted({t.upper() for t in FALLBACK_TICKERS
                      if t.upper() not in EXEC_EXCL})

    # 2. Datos
    print("\n2. Descargando datos...")
    df_raw = download_data(tickers)

    # 3. Features
    print("\n3. Calculando features...")
    df_f = compute_features(df_raw)
    if df_f.empty:
        _save({'system_active':False,'reason':'no_features','n_signals':0,'signals':[]})
        return

    regime  = compute_regime(df_f)
    zscore  = compute_zscore(df_f)
    latest  = df_f.index.get_level_values('date').max()
    z_today = float(zscore.loc[latest]) if latest in zscore.index else float('nan')

    print(f"   Fecha: {latest.date()}  |  Zscore: {z_today:+.3f}")

    if z_today != z_today or z_today <= 0:
        print(f"   ⚠ SISTEMA INACTIVO (zscore={z_today:.3f})")
        _save({'date':latest.strftime('%Y-%m-%d'),'zscore':round(z_today,3),
               'system_active':False,'reason':'zscore_inactive',
               'n_signals':0,'signals':[]})
        return

    # 4. Predecir
    print("\n4. Prediciendo...")
    today_mask = df_f.index.get_level_values('date') == latest
    df_today   = df_f[today_mask].copy()

    for fn in feat_names:
        if fn not in df_today.columns:
            df_today[fn] = 0.0

    import xgboost as xgb
    X        = df_today[feat_names].fillna(0).values.astype(float)
    dmatrix  = xgb.DMatrix(X, feature_names=feat_names)
    p_buy_arr= buy_model.predict(dmatrix)
    p_buy    = pd.Series(p_buy_arr, index=df_today.index)

    # 5. Filtros
    reg_today    = regime[today_mask]
    min_pbuy     = float(cfg.get('min_buy_proba', 0.50))
    block_neutral= bool(cfg.get('block_neutral', True))
    block_lowvol = bool(cfg.get('block_low_vol', True))
    top_n        = int(cfg.get('top_n_per_date', 5))

    sigs = pd.DataFrame({'p_buy':p_buy,'regime':reg_today}, index=df_today.index)

    # Thresholds por régimen
    def get_thr(reg):
        return float(buy_thr.get(str(reg), buy_thr.get('global', min_pbuy)))

    thr_series = reg_today.map(get_thr)
    sigs = sigs[sigs['p_buy'] >= thr_series]

    if block_neutral: sigs = sigs[sigs['regime'] != 'neutral']
    if block_lowvol:  sigs = sigs[sigs['regime'] != 'low_vol']

    tkrs = sigs.index.get_level_values('ticker')
    sigs = sigs[~tkrs.isin(EXEC_EXCL)]

    # Earnings
    blackout  = load_blackout()
    today_ts  = pd.Timestamp(latest).normalize()
    to_drop   = [(t,d) for t,d in sigs.index
                 if any((today_ts+pd.Timedelta(days=o)) in blackout.get(t,set())
                        for o in range(6))]
    sigs = sigs.drop(to_drop, errors='ignore')

    # Top N
    if top_n and len(sigs) > top_n:
        sigs = sigs.nlargest(top_n, 'p_buy')

    entry_date = latest.strftime('%Y-%m-%d')
    exit_date  = (latest + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    result = []
    for (tkr,_), row in sigs.iterrows():
        try:
            price = float(df_raw.xs(tkr,level='ticker')['close'].iloc[-1])
        except Exception:
            price = None
        result.append({
            'ticker':        tkr,
            'entry_date':    entry_date,
            'exit_expected': exit_date,
            'p_buy':         round(float(row['p_buy']),4),
            'regime':        str(row['regime']),
            'asset_type':    _cls(tkr),
            'entry_price':   round(price,2) if price else None,
            'zscore':        round(z_today,3),
        })
        print(f"   ✓ {tkr:<8} p_buy={row['p_buy']:.3f}  {row['regime']}")

    if not result:
        print("   Sin señales hoy tras filtros.")

    payload = {
        'generated_at':  now.isoformat()+'Z',
        'date':          entry_date,
        'zscore':        round(z_today,3),
        'system_active': True,
        'n_signals':     len(result),
        'signals':       result,
    }

    _save(payload)
    _update_log(payload)
    print(f"\n   ✓ {len(result)} señales guardadas en signals_hoy.json")

def _save(payload):
    os.makedirs(os.path.dirname(DOCS_JSON), exist_ok=True)
    for path in [OUT_JSON, DOCS_JSON]:
        with open(path,'w') as f:
            json.dump(payload, f, indent=2)

def _update_log(payload):
    log = []
    if os.path.exists(LOG_JSON):
        try:
            with open(LOG_JSON) as f: log = json.load(f)
        except Exception: log = []
    log.append(payload)
    log = log[-90:]
    with open(LOG_JSON,'w') as f:
        json.dump(log, f, indent=2)

if __name__ == '__main__':
    main()
