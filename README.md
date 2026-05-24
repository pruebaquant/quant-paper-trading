# Quant Paper Trading — HP=5d

Dashboard automático de paper trading para el sistema de momentum con filtro de régimen.

## Setup

Ver instrucciones completas en el chat.

## Estructura

```
model/
  trained_model.pkl     ← subir manualmente después de correr main_hp5.py
  earnings_cache.pkl    ← subir manualmente
daily_signals.py        ← corre en GitHub Actions cada día a las 16:00 ET
signals_hoy.json        ← generado automáticamente
signals_log.json        ← historial de señales (últimos 90 días)
docs/
  index.html            ← dashboard web (GitHub Pages)
  signals_hoy.json      ← copia para GitHub Pages
```

## Cómo funciona

GitHub Actions corre `daily_signals.py` cada día de lunes a viernes a las 21:00 UTC (16:00 ET).
El script descarga precios, calcula features, genera señales y actualiza el dashboard.
