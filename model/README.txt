Subir estos archivos después de correr main_hp5.py:

  trained_model.pkl    →  desde quant_pipeline/output_hp5/trained_model.pkl
  earnings_cache.pkl   →  desde quant_pipeline/output_hp5/earnings_cache.pkl

Comando para subir (desde la carpeta del repo):
  cp /ruta/a/quant_pipeline/output_hp5/trained_model.pkl model/
  cp /ruta/a/quant_pipeline/output_hp5/earnings_cache.pkl model/
  git add model/
  git commit -m "add trained model"
  git push
