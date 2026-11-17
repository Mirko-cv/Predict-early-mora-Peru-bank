# Predict-early-mora-Peru-bank

Scorecard de **morosidad temprana** sobre 130,184 clientes con información de buró (RCC), abril–noviembre 2021.
WoE + regresión logística, validación fuera de tiempo (oct–nov 2021) y app de gestión del modelo.

| Archivo | Qué es |
|---|---|
| `modelo_early_morosidad8-vf5.ipynb` | Proyecto completo: EDA, selección de variables, modelo, validación, decisión y exportación |
| `modelo_export/` | Resultados que deja el notebook; es lo único que lee la app |
| `app_gestion_modelo.py` | App Streamlit: desempeño, estabilidad, punto de corte, simulador y gobierno del modelo |

**Resultados:** Gini fuera de tiempo 0.403 · 14 variables · alertando al 20% de peor score se detecta al 43% de los futuros morosos.

## Ejecutar la app

```bash
pip install streamlit
streamlit run app_gestion_modelo.py
```

La app solo necesita Streamlit (que ya incluye pandas, numpy, altair y pyarrow). No lee la base original.
