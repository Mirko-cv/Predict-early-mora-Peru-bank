"""
Gestión del modelo · Scorecard early de morosidad
=================================================
Muestra EXCLUSIVAMENTE lo que deja la última celda del notebook en `modelo_export/`:
no lee la base original ni ningún otro archivo de datos.

    streamlit run app_gestion_modelo.py

Requisitos: solo streamlit (trae pandas, numpy, altair y pyarrow). El simulador usa scorecard.json,
el scorecard en formato plano, así que no necesita optbinning ni el pickle.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ═══════════════════════════════ Configuración ═══════════════════════════════
EXPORT = Path(__file__).resolve().parent / "modelo_export"
st.set_page_config(page_title="Gestión del modelo · Early morosidad", page_icon="◆", layout="wide")

# Paleta del deck (inspirada en Ripley, no copiada)
C = dict(plum="#2B1030", plum2="#3C1A42", plum3="#54285C", cream="#FAF6F8", cream2="#F1E7EE",
         card="#FFFBFD", ink="#241026", soft="#6E5A71", mist="#E2CFE4", magenta="#A42A7F",
         rose="#E0609A", amber="#C9821F", pale="#CE9BC2", line="#DCC9DD",
         ok="#2E8F6B", warn="#C9821F", bad="#C0392B")
SERIE = [C["magenta"], C["amber"], C["rose"], C["plum3"], C["pale"]]
MODELOS_ORDEN = ["Scorecard", "Logit", "Árbol prof.4", "LightGBM"]

# Umbrales de monitoreo (verde si cumple el primero, ámbar si cumple el segundo, rojo si no)
UMBRALES = {"gini": (0.36, 0.30, True), "ks": (0.30, 0.25, True),
            "psi": (0.10, 0.25, False), "calib": (0.10, 0.25, False)}


def estado(valor, clave):
    v, a, mayor = UMBRALES[clave]
    if mayor:
        return "ok" if valor >= v else "warn" if valor >= a else "bad"
    return "ok" if valor < v else "warn" if valor < a else "bad"


ETIQ_ESTADO = {"ok": "Estable", "warn": "Vigilar", "bad": "Alerta"}

# ═══════════════════════════════ Estilo ═══════════════════════════════
st.markdown(f"""
<style>
:root {{
  --plum:{C['plum']}; --plum2:{C['plum2']}; --cream:{C['cream']}; --cream2:{C['cream2']}; --card:{C['card']};
  --ink:{C['ink']}; --soft:{C['soft']}; --magenta:{C['magenta']}; --rose:{C['rose']}; --amber:{C['amber']};
  --line:{C['line']}; --ok:{C['ok']}; --warn:{C['warn']}; --bad:{C['bad']};
  --r-md: 8px; --sh-s: 0 1px 2px rgba(43,16,48,.06), 0 1px 3px rgba(43,16,48,.10);
}}
.stApp {{ background: var(--cream); color: var(--ink); }}
.block-container {{ padding-top: 3.6rem; max-width: 1280px; }}
[data-testid="stHeader"] {{ background: rgba(250,246,248,.85); backdrop-filter: blur(6px); }}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -.01em; }}
[data-testid="stSidebar"] {{ background: var(--plum); }}
[data-testid="stSidebar"] * {{ color: {C['mist']} !important; }}

/* Hero */
.hero {{ background: linear-gradient(120deg, var(--plum) 0%, var(--plum2) 60%, {C['plum3']} 100%);
  border-radius: var(--r-md); box-shadow: var(--sh-s); padding: 22px 26px; margin-bottom: 14px; color: {C['mist']}; }}
.hero .eyebrow {{ color: var(--rose); font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
.hero h1 {{ color: #FFF6FB; font-size: clamp(1.35rem, 2.6vw, 2rem); margin: 4px 0 6px; line-height: 1.15; }}
.hero p {{ margin: 0; font-size: .92rem; color: {C['mist']}; max-width: 880px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
.chip {{ font-size: .74rem; padding: 3px 10px; border-radius: 999px; border: 1px solid rgba(226,207,228,.35); color: #FFF6FB; }}
.chip.solid {{ background: var(--magenta); border-color: var(--magenta); }}

/* Encabezado de sección */
.sec {{ margin: 22px 0 10px; }}
.sec .eyebrow {{ color: var(--magenta); font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
.sec h2 {{ font-size: clamp(1.1rem, 2vw, 1.4rem); margin: 2px 0 4px; }}
.sec p {{ color: var(--soft); font-size: .88rem; margin: 0; max-width: 900px; }}

/* Cards */
.grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-bottom: 14px; }}
.grid.wide {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: var(--r-md);
  box-shadow: var(--sh-s); padding: 14px 16px; }}
.card.accent {{ border-left: 4px solid var(--magenta); }}
.card.dark {{ background: var(--plum2); border-color: {C['plum3']}; color: {C['mist']}; }}
.card.dark .kv {{ color: #FFF6FB; }}
.k-lab {{ font-size: .74rem; color: var(--soft); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.kv {{ font-size: clamp(1.4rem, 2.4vw, 1.9rem); font-weight: 750; color: var(--ink); line-height: 1.1; margin: 6px 0 4px; }}
.k-sub {{ font-size: .8rem; color: var(--soft); }}
.badge {{ display: inline-block; font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; }}
.badge.ok {{ background: rgba(46,143,107,.12); color: var(--ok); }}
.badge.warn {{ background: rgba(201,130,31,.14); color: var(--warn); }}
.badge.bad {{ background: rgba(192,57,43,.12); color: var(--bad); }}
.badge.neutral {{ background: var(--cream2); color: var(--soft); }}
.vname {{ font-weight: 700; font-size: .84rem; overflow-wrap: anywhere; }}
.vfam {{ font-size: .74rem; color: var(--soft); margin-bottom: 8px; }}
.bar {{ height: 6px; background: var(--cream2); border-radius: 999px; overflow: hidden; margin: 6px 0 8px; }}
.bar > span {{ display: block; height: 100%; background: var(--magenta); border-radius: 999px; }}
.row {{ display: flex; justify-content: space-between; font-size: .78rem; color: var(--soft); }}
.note {{ font-size: .86rem; color: var(--ink); line-height: 1.45; }}
.note b {{ color: var(--magenta); }}

/* Contenedores nativos con borde = cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-radius: var(--r-md) !important;
  border-color: var(--line) !important; box-shadow: var(--sh-s); }}

/* Botones */
.stButton > button, .stDownloadButton > button {{ border-radius: var(--r-md); border: 1px solid var(--line);
  box-shadow: var(--sh-s); font-weight: 600; }}
.stButton > button[kind="primary"] {{ background: var(--magenta); border-color: var(--magenta); color: #fff; }}
.stButton > button:hover {{ border-color: var(--magenta); color: var(--magenta); }}
.stButton > button[kind="primary"]:hover {{ background: {C['plum3']}; color: #fff; }}
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: var(--r-md);
  box-shadow: var(--sh-s); padding: 10px 14px; }}

button[data-selected="true"], button[aria-checked="true"][role="radio"] {{
  background: var(--magenta) !important; border-color: var(--magenta) !important; color: #fff !important; }}
button[data-selected="true"] *, button[aria-checked="true"][role="radio"] * {{ color: #fff !important; }}
[data-baseweb="slider"] [role="slider"] {{ background: var(--magenta) !important; }}
[data-testid="stSidebar"] code {{ background: var(--plum2); color: #FFF6FB !important; }}

@media (max-width: 640px) {{
  .hero {{ padding: 16px; }}
  .block-container {{ padding-left: .8rem; padding-right: .8rem; }}
}}
</style>
""", unsafe_allow_html=True)


def tema_altair():
    return {"config": {
        "background": "transparent", "view": {"stroke": None},
        "font": "Inter, system-ui, sans-serif",
        "axis": {"labelColor": C["soft"], "titleColor": C["soft"], "gridColor": C["cream2"],
                 "domainColor": C["line"], "tickColor": C["line"], "labelFontSize": 11, "titleFontSize": 11},
        "legend": {"labelColor": C["ink"], "titleColor": C["soft"], "orient": "bottom"},
        "range": {"category": SERIE},
        "title": {"color": C["ink"], "fontSize": 13, "anchor": "start", "fontWeight": 700}}}


alt.themes.register("ripley", tema_altair)
alt.themes.enable("ripley")


# ═══════════════════════════════ Componentes ═══════════════════════════════
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def seccion(eyebrow, titulo, texto=""):
    html(f'<div class="sec"><div class="eyebrow">{eyebrow}</div><h2>{titulo}</h2>'
         f'{f"<p>{texto}</p>" if texto else ""}</div>')


def kpi(lab, valor, sub="", est=None, clase=""):
    badge = f'<span class="badge {est}">{ETIQ_ESTADO.get(est, est)}</span>' if est else ""
    return (f'<div class="card {clase}"><div class="k-lab">{lab}</div><div class="kv">{valor}</div>'
            f'<div class="k-sub">{badge} {sub}</div></div>')


def grid(items, clase=""):
    html(f'<div class="grid {clase}">{"".join(items)}</div>')


def grafico(ch):
    st.altair_chart(ch.properties(height=ch.to_dict().get("height", 280)), width="stretch")


# ═══════════════════════════════ Carga (solo modelo_export/) ═══════════════════════════════
def _json(ruta):
    for enc in ("utf-8", "cp1252"):
        try:
            return json.load(open(ruta, encoding=enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(ruta)


@st.cache_resource(show_spinner="Cargando scorecard…")
def cargar_modelo():
    return _json(EXPORT / "scorecard.json")


def puntos_de(var, x):
    """Puntos que aporta el valor x de una variable, con los cortes exactos del binning."""
    d = MOD["variables"][var]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return d["puntos_missing"]
    if d["tipo"] == "numerical":
        return d["puntos"][int(np.searchsorted(d["cortes"], x, side="right"))]
    for i, grupo in enumerate(d["categorias"]):
        if str(x) in grupo:
            return d["puntos"][i]
    return d["puntos_missing"]


@st.cache_data(show_spinner="Cargando resultados…")
def cargar():
    leer = lambda n, **kw: pd.read_csv(EXPORT / n, **kw)
    puntos = leer("puntos.csv")
    # exports de Windows + pandas 3 guardaron el atributo categórico como repr de un arreglo
    puntos["atributo"] = (puntos["atributo"].astype(str)
                          .str.replace(r"<ArrowStringArray>\n\[(.*?)\]\n.*", r"\1", regex=True)
                          .str.replace("'", ""))
    cli = pd.read_parquet(EXPORT / "clientes.parquet")
    cli["banda"] = cli["banda"].astype(str)
    met = leer("metricas.csv")
    met[["Modelo", "Muestra"]] = met["modelo"].str.split(" · ", expand=True)
    return {"cli": cli, "vars": leer("variables.csv", index_col=0), "puntos": puntos, "met": met,
            "deciles": leer("deciles.csv", index_col=0), "psi_mes": leer("psi_mensual.csv", index_col=0),
            "psi_var": leer("psi_variables.csv", index_col=0), "reglas": leer("reglas.csv", index_col=0),
            "auditoria": leer("auditoria_reglas.csv", index_col=0), "matriz": leer("matriz_decision.csv", index_col=0),
            "sens": leer("sensibilidad_decision.csv", index_col=0), "politica": leer("politica.csv", index_col=0),
            "gaps": leer("gap_folds.csv"), "ficha": _json(EXPORT / "ficha_modelo.json")}


if not EXPORT.exists():
    st.error("No encuentro `modelo_export/` junto a este archivo. Ejecuta la última celda del notebook para generarla.")
    st.stop()

D = cargar()
CLI, FICHA, VARS = D["cli"], D["ficha"], D["vars"]
OOT = CLI[CLI.split == "OOT"]
BANDAS = ["E · muy alto", "D · alto", "C · medio", "B · bajo", "A · muy bajo"]


def met(modelo, muestra, col):
    f = D["met"][(D["met"].Modelo == modelo) & (D["met"].Muestra == muestra)]
    return float(f[col].iloc[0]) if len(f) else np.nan


def captura(df, pct):
    corte = df.score.quantile(pct)
    return df.bad[df.score <= corte].sum() / df.bad.sum()


# ═══════════════════════════════ Páginas ═══════════════════════════════
def pag_resumen():
    g = FICHA["gini"]
    calib = FICHA["calibracion_oot_obs_pred"]
    html(f"""<div class="hero"><div class="eyebrow">Scorecard early de morosidad · v{FICHA.get('version', '')}</div>
      <h1>El modelo ordena bien fuera de tiempo; el nivel de PD requiere recalibración</h1>
      <p>{FICHA['n_variables']} variables de buró, WoE + regresión logística. Desarrollo {FICHA['ventana_desarrollo']},
      validación fuera de tiempo {FICHA['ventana_oot']} sobre {len(OOT):,} clientes.</p>
      <div class="chips"><span class="chip solid">Champion: Scorecard</span><span class="chip">Challenger: LightGBM</span>
      <span class="chip">Reglas: {', '.join(FICHA['reglas_aplicadas'])}</span>
      <span class="chip">SAMPLE_FRAC = {FICHA['sample_frac']}</span></div></div>""")

    grid([kpi("Gini fuera de tiempo", f"{g['OOT']:.3f}", f"TRAIN {g['TRAIN']:.3f} · TEST {g['TEST']:.3f}", estado(g["OOT"], "gini"), "accent"),
          kpi("KS fuera de tiempo", f"{FICHA['ks_oot']:.3f}", "umbral 0.25", estado(FICHA["ks_oot"], "ks")),
          kpi("PSI del score", f"{FICHA['psi_score_oot']:.3f}", "TRAIN → OOT", estado(FICHA["psi_score_oot"], "psi")),
          kpi("Calibración obs/pred", f"{calib:.2f}", "1.00 = perfecto", estado(abs(calib - 1), "calib")),
          kpi("Mora capturada", f"{captura(OOT, .2):.0%}", "alertando al 20% peor"),
          kpi("Ventaja del challenger", f"{FICHA['challenger_gini_oot'] - g['OOT']:+.3f}", "Gini OOT de LightGBM")], "wide")

    seccion("Lectura", "Qué dice el tablero")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        d = D["met"][D["met"].Modelo.isin(MODELOS_ORDEN)].assign(Gini=lambda x: x.Gini.round(3))
        grafico(alt.Chart(d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Modelo:N", sort=MODELOS_ORDEN, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Muestra:N", sort=["TRAIN", "TEST", "OOT"]),
            y=alt.Y("Gini:Q"),
            color=alt.Color("Muestra:N", sort=["TRAIN", "TEST", "OOT"], title=None,
                            scale=alt.Scale(range=[C["pale"], C["magenta"], C["amber"]])),
            tooltip=["Modelo", "Muestra", "Gini"]).properties(height=240, title="Gini por modelo y muestra"))
    with c2:
        mora = (OOT.groupby("banda").bad.agg(["size", "mean"]).reindex(BANDAS).reset_index()
                .rename(columns={"size": "clientes", "mean": "mora"}))
        ch = alt.Chart(mora).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("banda:N", sort=BANDAS, title=None), x=alt.X("mora:Q", axis=alt.Axis(format="%"), title="mora observada"),
            color=alt.Color("banda:N", sort=BANDAS, legend=None,
                            scale=alt.Scale(range=[C["magenta"], C["rose"], C["pale"], C["line"], C["cream2"]])),
            tooltip=["banda", "clientes", alt.Tooltip("mora:Q", format=".2%")]
        ).properties(height=220, title="Mora por banda · fuera de tiempo")
        grafico(ch)

    html(f"""<div class="grid wide">
      <div class="card accent note"><b>Discrimina igual fuera de tiempo.</b> El Gini se mueve menos de un punto entre
      entrenamiento, prueba y OOT: no hay sobreajuste y el ordenamiento se sostiene en meses que el modelo no vio.</div>
      <div class="card accent note"><b>Subestima el nivel.</b> La mora real OOT supera a la predicha en un {calib - 1:.0%}.
      Es un cambio de nivel de la cartera, no de forma: no afecta bandas ni alertas, sí cualquier uso de la PD como número.</div>
      <div class="card accent note"><b>LightGBM es challenger, no champion.</b> Gana {FICHA['challenger_gini_oot'] - g['OOT']:.3f} de Gini,
      pero con una brecha entre entrenamiento y prueba mucho mayor y sin tabla de puntos explicable.</div></div>""")


def pag_variables():
    seccion("Variables del modelo", f"{len(VARS)} variables, {VARS.familia.nunique()} dimensiones de riesgo",
            "Peso = rango de puntos que la variable puede sumar o restar. PSI = estabilidad TRAIN → OOT.")
    vmax = VARS.peso_puntos.max()
    tarjetas = []
    for v, r in VARS.sort_values("peso_puntos", ascending=False).iterrows():
        e = estado(r.PSI_TRAIN_OOT, "psi")
        tarjetas.append(f"""<div class="card"><div class="vname">{v}</div><div class="vfam">{r.familia}</div>
          <div class="row"><span>Peso</span><b style="color:{C['ink']}">{r.peso_puntos:.1f} pts</b></div>
          <div class="bar"><span style="width:{r.peso_puntos / vmax * 100:.0f}%"></span></div>
          <div class="row"><span>IV {r.IV:.3f}</span><span>VIF {r.VIF:.2f}</span></div>
          <div class="row" style="margin-top:4px"><span>faltante {r.pct_faltante:.0%}</span>
          <span class="badge {e}">PSI {r.PSI_TRAIN_OOT:.3f}</span></div></div>""")
    grid(tarjetas)

    seccion("Detalle", "Tramos, tasa de malos y puntos")
    orden = VARS.sort_values("peso_puntos", ascending=False).index.tolist()
    v = st.pills("Variable", orden, default=orden[0], label_visibility="collapsed") or orden[0]
    t = D["puntos"][D["puntos"].variable == v].reset_index(drop=True)
    t["orden"] = range(len(t))
    r = VARS.loc[v]
    grid([kpi("Familia", r.familia.split(":")[0], r.familia),
          kpi("Information Value", f"{r.IV:.3f}"),
          kpi("Coeficiente", f"{r.coeficiente:.3f}", f"p = {r.p_valor:.1e}"),
          kpi("PSI TRAIN → OOT", f"{r.PSI_TRAIN_OOT:.3f}", "", estado(r.PSI_TRAIN_OOT, "psi"))])
    c1, c2 = st.columns(2)
    base = alt.Chart(t).encode(x=alt.X("atributo:N", sort=alt.SortField("orden"), title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)))
    with c1:
        grafico((base.mark_bar(color=C["magenta"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("tasa_malos:Q", axis=alt.Axis(format="%"), title="tasa de malos"),
            tooltip=["atributo", "n", alt.Tooltip("tasa_malos:Q", format=".2%")])
            + base.mark_rule(color=C["amber"], strokeDash=[4, 3]).encode(y=alt.datum(float((t.tasa_malos * t.n).sum() / t.n.sum()))))
            .properties(height=260, title="Tasa de malos por tramo (línea = promedio)"))
    with c2:
        grafico(base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("puntos:Q", scale=alt.Scale(zero=False), title="puntos"),
            color=alt.Color("puntos:Q", scale=alt.Scale(range=[C["pale"], C["plum"]]), legend=None),
            tooltip=["atributo", "puntos", alt.Tooltip("WoE:Q", format=".3f")]).properties(height=260, title="Puntos que aporta cada tramo"))
    with st.expander("Tabla de puntos"):
        st.dataframe(t.drop(columns=["variable", "orden"]).style.format(
            {"n": "{:,}", "%clientes": "{:.1%}", "tasa_malos": "{:.2%}", "WoE": "{:.3f}", "puntos": "{:.1f}"}),
            hide_index=True, width="stretch")


def pag_desempeno():
    seccion("Desempeño", "Cuatro modelos, tres muestras", "Gini, KS y Average Precision no dependen de un corte; "
            "Recall/Precision/F1 se miden alertando al 20% de mayor PD.")
    muestra = st.segmented_control("Muestra", ["TRAIN", "TEST", "OOT"], default="OOT") or "OOT"
    t = (D["met"][D["met"].Muestra == muestra].set_index("Modelo").reindex(MODELOS_ORDEN)
         .drop(columns=["modelo", "Muestra"]))
    grid([kpi(m, f"{t.loc[m, 'Gini']:.3f}", f"KS {t.loc[m, 'KS']:.3f} · AP {t.loc[m, 'AveragePrecision']:.3f}",
              estado(t.loc[m, "Gini"], "gini"), "accent" if m == "Scorecard" else "") for m in MODELOS_ORDEN])
    with st.container(border=True):
        st.dataframe(t.style.format({"n": "{:,}", **{c: "{:.4f}" for c in t.columns if c != "n"}})
                     .highlight_max(subset=["Gini", "KS", "AveragePrecision"], color="#F3DCEB"),
                     width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        dec = D["deciles"].reset_index()
        b = alt.Chart(dec).encode(x=alt.X("decil:O", title="decil de score (1 = más riesgoso)", axis=alt.Axis(labelAngle=0)))
        grafico(alt.layer(
            b.mark_bar(color=C["magenta"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                y=alt.Y("tasa:Q", axis=alt.Axis(format="%"), title="mora del decil"), tooltip=["decil", alt.Tooltip("tasa:Q", format=".2%"), alt.Tooltip("lift:Q", format=".2f")]),
            b.mark_line(color=C["amber"], point=True).encode(
                y=alt.Y("%malos_acum:Q", axis=alt.Axis(format="%", titleColor=C["amber"]), title="malos acumulados"))
        ).resolve_scale(y="independent").properties(height=280, title="Scorecard · deciles fuera de tiempo"))
    with c2:
        gp = D["gaps"]
        grafico(alt.Chart(gp).mark_line(point=True).encode(
            x=alt.X("Fold:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("Gap_AUC:Q", title="AUC train − AUC validación"),
            color=alt.Color("Modelo:N", scale=alt.Scale(range=[C["magenta"], C["amber"]])),
            tooltip=["Modelo", "Fold", alt.Tooltip("Gap_AUC:Q", format=".4f")]
        ).properties(height=280, title="Sobreajuste por fold (validación cruzada)"))


def pag_estabilidad():
    seccion("Monitoreo", "Estabilidad y calibración", "Semáforo PSI: < 0.10 estable · 0.10–0.25 vigilar · > 0.25 alerta.")
    pm = D["psi_mes"].reset_index().rename(columns={"index": "periodo", D["psi_mes"].index.name or "index": "periodo"})
    pm.columns = ["periodo", "PSI"]
    pm["periodo"] = pm.periodo.astype(str)
    pm["tramo"] = np.where(pm.periodo.isin(["202110", "202111"]), "OOT", "desarrollo")
    calib = (CLI.groupby("PERIODO").agg(PD=("PD", "mean"), real=("bad", "mean")).reset_index()
             .melt("PERIODO", var_name="serie", value_name="tasa").assign(PERIODO=lambda d: d.PERIODO.astype(str)))
    oot_obs = OOT.bad.mean() / OOT.PD.mean()
    grid([kpi("PSI máx. mensual", f"{pm.PSI.max():.3f}", f"en {pm.loc[pm.PSI.idxmax(), 'periodo']}", estado(pm.PSI.max(), "psi")),
          kpi("PSI máx. variable", f"{D['psi_var'].PSI.max():.3f}", D["psi_var"].PSI.idxmax(), estado(D["psi_var"].PSI.max(), "psi")),
          kpi("obs/pred OOT", f"{oot_obs:.2f}", f"real {OOT.bad.mean():.2%} · predicha {OOT.PD.mean():.2%}", estado(abs(oot_obs - 1), "calib"))])
    c1, c2 = st.columns(2)
    with c1:
        reglas = alt.Chart(pd.DataFrame({"y": [.10, .25]})).mark_rule(strokeDash=[4, 3], color=C["soft"]).encode(y="y:Q")
        grafico((alt.Chart(pm).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("periodo:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("PSI:Q", title="PSI del score vs TRAIN"),
            color=alt.Color("tramo:N", scale=alt.Scale(range=[C["pale"], C["amber"]])),
            tooltip=["periodo", alt.Tooltip("PSI:Q", format=".4f")]) + reglas).properties(height=270, title="PSI del score por mes"))
    with c2:
        grafico(alt.Chart(calib).mark_line(point=True, strokeWidth=2.4).encode(
            x=alt.X("PERIODO:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tasa:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["PD", "real"], range=[C["magenta"], C["amber"]]),
                            legend=alt.Legend(title=None, labelExpr="datum.label == 'PD' ? 'PD media predicha' : 'mora real'")),
            tooltip=["PERIODO", "serie", alt.Tooltip("tasa:Q", format=".2%")]
        ).properties(height=270, title="Calibración por mes: predicha vs real"))

    pv = D["psi_var"].reset_index()
    pv.columns = ["variable", "PSI"]
    pv["estado"] = pv.PSI.map(lambda x: ETIQ_ESTADO[estado(x, "psi")])
    grafico(alt.Chart(pv).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("variable:N", sort="-x", title=None), x=alt.X("PSI:Q", title="PSI TRAIN → OOT"),
        color=alt.Color("estado:N", scale=alt.Scale(domain=list(ETIQ_ESTADO.values()), range=[C["ok"], C["warn"], C["bad"]])),
        tooltip=["variable", alt.Tooltip("PSI:Q", format=".4f")]).properties(height=34 * len(pv) // 2 + 120, title="PSI por variable"))
    html('<div class="card accent note"><b>Cómo se lee.</b> Si el PSI del score se mantiene bajo pero la calibración se '
         'separa, la cartera cambió de nivel y no de composición: se recalibra el intercepto con la ventana reciente. Si el '
         'PSI de una variable pasa a rojo, se revisa esa variable antes que el modelo. <code>INT_SALDO</code> tiene un '
         'disparador adicional: un cambio en la política de precios obliga a revalidar.</div>')


def pag_corte():
    seccion("Punto de corte", "A cuántos clientes alertar",
            "Los cortes se calculan sobre la cartera fuera de tiempo. El modelo da la curva; negocio pone los supuestos.")
    n, malos = len(OOT), int(OOT.bad.sum())
    filas = []
    for pct in (.10, .20, .30, .40, .50):
        al = OOT.score <= OOT.score.quantile(pct)
        filas.append((f"{pct:.0%} peor", al))
    filas += [("Banda E", OOT.banda.eq(BANDAS[0])), ("Bandas E + D", OOT.banda.isin(BANDAS[:2]))]
    cortes = pd.DataFrame([{"corte": k, "% cartera": a.mean(), "alertados": int(a.sum()),
                            "malos capturados": int(OOT.bad[a].sum()), "recall": OOT.bad[a].sum() / malos,
                            "mora alertados": OOT.bad[a].mean(), "mora no alertados": OOT.bad[~a].mean()} for k, a in filas])

    with st.container(border=True):
        st.markdown("**Supuestos de negocio**")
        presets = {"Conservador": (25, 500, .20), "Base": (15, 600, .30), "Agresivo": (8, 800, .40)}
        b = st.columns(len(presets))
        for (nombre, vals), col in zip(presets.items(), b):
            if col.button(nombre, width="stretch",
                          type="primary" if st.session_state.get("preset", "Base") == nombre else "secondary"):
                st.session_state.update(preset=nombre, costo=vals[0], perdida=vals[1], efect=vals[2])
                st.rerun()
        for k, v in zip(("costo", "perdida", "efect"), presets["Base"]):
            st.session_state.setdefault(k, v)
        c = st.columns(3)
        costo = c[0].number_input("Costo por cliente gestionado (S/)", 1, 500, key="costo")
        perdida = c[1].number_input("Pérdida por moroso early (S/)", 50, 10000, step=50, key="perdida")
        efect = c[2].slider("Pérdida evitable con alerta", .05, .90, step=.05, key="efect")

    cortes["beneficio neto"] = cortes["malos capturados"] * perdida * efect - cortes["alertados"] * costo
    equilibrio = costo / (perdida * efect)
    mejor = cortes.loc[cortes["beneficio neto"].idxmax()]
    grid([kpi("Corte recomendado", mejor.corte, f"{mejor['% cartera']:.0%} de la cartera", clase="accent"),
          kpi("Morosos capturados", f"{mejor.recall:.0%}", f"{int(mejor['malos capturados']):,} de {malos:,}"),
          kpi("Beneficio neto", f"S/ {mejor['beneficio neto'] / 1000:,.0f}k", "en los 2 meses OOT"),
          kpi("Mora de equilibrio", f"{equilibrio:.1%}", "se amplía mientras el tramo siguiente la supere")])
    d = cortes.assign(mejor=cortes.corte.eq(mejor.corte))
    grafico(alt.Chart(d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("corte:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False, labelFontSize=10)),
        y=alt.Y("beneficio neto:Q", title="S/"),
        color=alt.condition("datum.mejor", alt.value(C["amber"]), alt.value(C["magenta"])),
        tooltip=["corte", alt.Tooltip("beneficio neto:Q", format=",.0f"), alt.Tooltip("recall:Q", format=".0%")]
    ).properties(height=260, title="Beneficio neto por corte (ámbar = recomendado)"))
    st.dataframe(cortes.set_index("corte")[["% cartera", "recall", "mora alertados", "mora no alertados", "beneficio neto"]]
                 .style.format({"% cartera": "{:.0%}", "recall": "{:.0%}", "mora alertados": "{:.1%}",
                                "mora no alertados": "{:.1%}", "beneficio neto": "S/ {:,.0f}"})
                 .background_gradient(subset=["beneficio neto"], cmap="RdPu"), width="stretch")
    html(f'<div class="card accent note"><b>Recomendación con estos supuestos: {mejor.corte}.</b> Como ya son clientes, '
         f'el modelo no rechaza: alerta. Se prioriza capturar morosos (recall) sujeto a la capacidad de cobranza. Si la '
         f'capacidad no alcanza para {mejor["% cartera"]:.0%} de la cartera, la banda E es el corte operativo mínimo.</div>')


def pag_simulador():
    seccion("Simulador", "Puntuar un cliente", "Aplica la tabla de puntos exportada (scorecard.json). "
            "“Sin dato” cae en el tramo Missing, igual que en el modelo.")
    global MOD
    MOD = cargar_modelo()
    SEL, ESC = MOD["seleccion"], MOD["escala"]
    F, O = ESC["factor"], ESC["offset"]

    if st.button("Restablecer a cliente típico"):
        for k in [k for k in st.session_state if k.startswith(("s_", "n_", "na_"))]:
            del st.session_state[k]
        st.rerun()
    valores = {}
    with st.container(border=True):
        cols = st.columns(3)
        for i, v in enumerate(SEL):
            with cols[i % 3]:
                if v in MOD["categoricas"]:
                    ops = MOD["niveles"][v] + ["(sin dato)"]
                    x = st.selectbox(v, ops, index=ops.index(MOD.get("moda", {}).get(v, ops[0])), key=f"s_{v}")
                    valores[v] = np.nan if x == "(sin dato)" else x
                else:
                    r = MOD["rangos"][v]
                    sin = st.session_state.get(f"na_{v}", False)
                    x = st.number_input(v, value=float(round(r["mediana"], 2)), key=f"n_{v}", disabled=sin)
                    st.toggle("sin dato", key=f"na_{v}", help=f"{r['pct_faltante']:.0%} de la cartera no tiene este dato")
                    valores[v] = np.nan if sin else x

    pts = {v: puntos_de(v, valores[v]) for v in SEL}
    score = sum(pts.values())
    pd_ = 1 / (1 + np.exp((score - O) / F))
    banda = pd.cut([score], [-np.inf, *MOD["cortes_banda"], np.inf], labels=MOD["etiquetas_banda"])[0]
    base = MOD["puntos_neutro"]          # lo que aporta una variable con WoE = 0 (cliente neutro)
    grid([kpi("Score", f"{score:.0f}", "", None, "dark"),
          kpi("PD estimada", f"{pd_:.2%}", "nivel de desarrollo"),
          kpi("Banda", str(banda), "alertar" if banda in BANDAS[:2] else "sin gestión",
              "bad" if banda == BANDAS[0] else "warn" if banda == BANDAS[1] else None),
          kpi("Percentil en la cartera", f"{(CLI.score < score).mean():.0%}", "0% = más riesgoso")])
    ap = pd.DataFrame({"variable": SEL, "aporte": [pts[v] - base for v in SEL]})
    ap["sentido"] = np.where(ap.aporte >= 0, "suma (menos riesgo)", "resta (más riesgo)")
    grafico(alt.Chart(ap).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("variable:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("aporte:Q", title="puntos frente a un cliente neutro"),
        color=alt.Color("sentido:N", scale=alt.Scale(domain=["suma (menos riesgo)", "resta (más riesgo)"], range=[C["ok"], C["magenta"]])),
        tooltip=["variable", alt.Tooltip("aporte:Q", format="+.1f")]).properties(height=28 * len(SEL) + 40, title="Por qué este score"))


def pag_gobierno():
    seccion("Gobierno", "Decisiones documentadas", "Lo que un validador independiente revisa: reglas, criterio de "
            "elección del modelo y costo de las exclusiones de política.")
    grid([kpi("Reglas aplicadas", str(len(FICHA["reglas_aplicadas"])), "; ".join(FICHA["reglas_aplicadas"]), None, "accent"),
          kpi("Reglas descartadas", str(len(FICHA["reglas_descartadas"])), "verificadas con datos"),
          kpi("Excluidas por política", str(len(FICHA["excluidas_por_politica"])), ", ".join(FICHA["excluidas_por_politica"][:3]) + "…"),
          kpi("Costo de la política", f"{D['politica'].iloc[-1, -1]:+.3f}", "Gini OOT que se resigna")])

    tabs = st.tabs(["Reglas de negocio", "Elección del modelo", "Política", "Ficha"])
    with tabs[0]:
        r = D["reglas"]
        st.dataframe(r.style.format({c: "{:.4f}" for c in r.select_dtypes("float").columns})
                     .highlight_max(subset=[c for c in r.columns if "Gini" in c], color="#F3DCEB"), width="stretch")
        html('<div class="grid wide">' + "".join(f'<div class="card note">{t}</div>' for t in FICHA["reglas_descartadas"]) + "</div>")
        with st.expander("Auditoría variable por variable"):
            st.dataframe(D["auditoria"], width="stretch")
    with tabs[1]:
        m = D["matriz"].sort_values("TOTAL", ascending=False)
        st.dataframe(m.style.format("{:.2f}").background_gradient(subset=["TOTAL"], cmap="RdPu"), width="stretch")
        st.caption("Sensibilidad: ¿cambia el ganador si cambian los pesos?")
        s = D["sens"]
        st.dataframe(s.style.format("{:.2f}").highlight_max(color="#F3DCEB"), width="stretch")
        html(f'<div class="card accent note"><b>Ganador por escenario:</b> '
             f'{", ".join(f"{k} → {v}" for k, v in s.idxmax().items())}. La recomendación descansa en interpretabilidad y '
             f'gobierno, no en superioridad estadística.</div>')
    with tabs[2]:
        st.dataframe(D["politica"].style.format("{:.4f}"), width="stretch")
        html('<div class="card note">Excluir género, estado civil y ubicación es una decisión de política, no un requisito '
             'estadístico: se declara y se mide su costo. Pendiente: medir tasa de alerta y obs/pred por grupo con Cumplimiento.</div>')
    with tabs[3]:
        st.json(FICHA, expanded=False)


# ═══════════════════════════════ Navegación ═══════════════════════════════
PAGINAS = {"Resumen": pag_resumen, "Variables": pag_variables, "Desempeño": pag_desempeno,
           "Estabilidad": pag_estabilidad, "Punto de corte": pag_corte, "Simulador": pag_simulador, "Gobierno": pag_gobierno}

with st.sidebar:
    st.markdown("### ◆ Gestión del modelo")
    st.caption(f"Fuente: `modelo_export/` · {len(CLI):,} clientes puntuados")
    st.caption("Este tablero no lee la base original.")

pagina = st.pills("Sección", list(PAGINAS), default="Resumen", label_visibility="collapsed") or "Resumen"
PAGINAS[pagina]()
