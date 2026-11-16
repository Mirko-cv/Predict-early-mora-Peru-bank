"""
Aplicativo de gestión del modelo — Scorecard early de morosidad
===============================================================
Explora el resultado del modelo final, monitorea su salud y puntúa clientes nuevos.

Uso:   streamlit run app_gestion_modelo.py
Lee la carpeta modelo_export/ que genera la última celda de modelo_early_morosidad.ipynb.
Si el notebook se reejecuta con SAMPLE_FRAC = 1.0, el aplicativo refleja el nuevo modelo sin cambios.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────── Configuración ───────────────────────────
RAIZ = Path(__file__).resolve().parent
EXPORT = RAIZ / "modelo_export"

st.set_page_config(page_title="Gestión del modelo · Early morosidad", page_icon="📉", layout="wide")

CIRUELA, MAGENTA, ROSA, AMBAR, GRIS = "#2B1030", "#A42A7F", "#E0609A", "#C9821F", "#8aa6c1"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titleweight": "bold"})

# Umbrales de monitoreo: verde / ámbar / rojo
UMBRALES = {
    "gini":    {"verde": 0.36, "ambar": 0.30, "mayor_mejor": True},
    "ks":      {"verde": 0.30, "ambar": 0.25, "mayor_mejor": True},
    "psi":     {"verde": 0.10, "ambar": 0.25, "mayor_mejor": False},
    "calib":   {"verde": 0.10, "ambar": 0.25, "mayor_mejor": False},  # |obs/pred - 1|
}


def semaforo(valor, clave):
    u = UMBRALES[clave]
    if u["mayor_mejor"]:
        return ("🟢", "verde") if valor >= u["verde"] else ("🟡", "ámbar") if valor >= u["ambar"] else ("🔴", "rojo")
    return ("🟢", "verde") if valor < u["verde"] else ("🟡", "ámbar") if valor < u["ambar"] else ("🔴", "rojo")


# ─────────────────────────── Carga ───────────────────────────
def _leer_json(ruta):
    """El JSON se escribe en utf-8; si viene de una corrida anterior en Windows puede estar en cp1252."""
    for enc in ("utf-8", "cp1252"):
        try:
            return json.load(open(ruta, encoding=enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta}")


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    return pickle.load(open(EXPORT / "modelo.pkl", "rb"))


@st.cache_data(show_spinner="Cargando resultados…")
def cargar_datos():
    leer = lambda n, **kw: pd.read_csv(EXPORT / n, **kw)
    return {
        "clientes": pd.read_parquet(EXPORT / "clientes.parquet"),
        "variables": leer("variables.csv", index_col=0),
        "puntos": leer("puntos.csv"),
        "metricas": leer("metricas.csv"),
        "deciles": leer("deciles.csv", index_col=0),
        "cortes": leer("cortes.csv"),
        "psi_mensual": leer("psi_mensual.csv", index_col=0),
        "psi_variables": leer("psi_variables.csv", index_col=0),
        "reglas": leer("reglas.csv", index_col=0),
        "auditoria": leer("auditoria_reglas.csv", index_col=0),
        "matriz": leer("matriz_decision.csv", index_col=0),
        "sensibilidad": leer("sensibilidad_decision.csv", index_col=0),
        "politica": leer("politica.csv", index_col=0),
        "gaps": leer("gap_folds.csv"),
        "ficha": _leer_json(EXPORT / "ficha_modelo.json"),
    }


if not EXPORT.exists():
    st.error(f"No encuentro la carpeta `{EXPORT.name}/`. Ejecuta la última celda del notebook "
             f"`modelo_early_morosidad.ipynb` para generarla.")
    st.stop()

MOD, D = cargar_modelo(), cargar_datos()
SEL = MOD["seleccion"]
COEF, ESCALA = MOD["coeficientes"], MOD["escala"]
FACTOR, OFFSET, N_V = ESCALA["factor"], ESCALA["offset"], len(SEL)
CATS, NIVELES, RANGOS = MOD["categoricas"], MOD["niveles"], MOD["rangos"]
BANDAS, CORTES_BANDA = MOD["etiquetas_banda"], MOD["cortes_banda"]
CLI = D["clientes"]


# ─────────────────────────── Motor de scoring ───────────────────────────
def a_woe(d):
    """Aplica el binning guardado y devuelve la matriz WoE de las variables del modelo."""
    out = {}
    for v in SEL:
        b = MOD["bins"][v]
        x = d[v].values if b.dtype == "numerical" else d[v].astype(object).where(d[v].notna(), None).values
        out[v] = b.transform(x, metric="woe", metric_missing="empirical", metric_special="empirical")
    return pd.DataFrame(out, index=d.index)


def puntuar(d):
    """Devuelve PD, score, banda y los puntos que aporta cada variable."""
    W = a_woe(d)
    pts = pd.DataFrame({v: -(COEF[v] * W[v] + COEF["const"] / N_V) * FACTOR + OFFSET / N_V for v in SEL},
                       index=d.index)
    r = pd.DataFrame(index=d.index)
    r["score"] = pts.sum(axis=1)
    odds = np.exp((r["score"] - OFFSET) / FACTOR)
    r["PD"] = 1 / (1 + odds)
    r["banda"] = pd.cut(r["score"], [-np.inf, *CORTES_BANDA, np.inf], labels=BANDAS)
    return r, pts, W


PUNTOS_BASE = OFFSET / N_V - (COEF["const"] / N_V) * FACTOR      # puntos de un cliente con WoE = 0


def psi(esperado, actual, cortes=None):
    e, a = np.asarray(pd.Series(esperado).dropna()), np.asarray(pd.Series(actual).dropna())
    if cortes is None:
        cortes = np.unique(np.quantile(e, np.linspace(0, 1, 11)))
        cortes[0], cortes[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, cortes)[0] / len(e), 1e-4, None)
    pa = np.clip(np.histogram(a, cortes)[0] / len(a), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def limpiar_como_base(df):
    """Replica la limpieza del notebook para un CSV cargado por el usuario."""
    df = df.copy()
    for c in CATS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    for v in SEL:
        if v not in CATS and v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
    return df


def metrica_de(nombre_fila, col):
    f = D["metricas"][D["metricas"].modelo == nombre_fila]
    return float(f[col].iloc[0]) if len(f) else np.nan


# ═══════════════════════════ PÁGINAS ═══════════════════════════
def pag_salud():
    st.title("📉 Salud del modelo")
    st.caption(f"{D['ficha']['nombre']} · versión {D['ficha']['version']} · "
               f"{D['ficha']['n_variables']} variables · desarrollo {D['ficha']['ventana_desarrollo']} · "
               f"fuera de tiempo {D['ficha']['ventana_oot']}")

    gini_oot = D["ficha"]["gini"]["OOT"]
    ks_oot = D["ficha"]["ks_oot"]
    psi_oot = D["ficha"]["psi_score_oot"]
    calib = D["ficha"]["calibracion_oot_obs_pred"]

    st.subheader("Indicadores con semáforo")
    c = st.columns(4)
    for col, (titulo, valor, clave, fmt, ayuda) in zip(c, [
            ("Gini fuera de tiempo", gini_oot, "gini", "{:.3f}", "Capacidad de ordenar. Verde ≥ 0.36, rojo < 0.30."),
            ("KS fuera de tiempo", ks_oot, "ks", "{:.3f}", "Máxima separación entre acumuladas. Verde ≥ 0.30."),
            ("PSI del score", psi_oot, "psi", "{:.3f}", "Estabilidad de la población. Verde < 0.10, rojo > 0.25."),
            ("Calibración obs/pred", calib, "calib", "{:.3f}", "1.0 sería perfecto. Se evalúa |obs/pred − 1|.")]):
        v = abs(valor - 1) if clave == "calib" else valor
        icono, etiqueta = semaforo(v, clave)
        col.metric(f"{icono} {titulo}", fmt.format(valor), etiqueta, delta_color="off", help=ayuda)

    st.divider()
    izq, der = st.columns([1.3, 1])
    with izq:
        st.subheader("Diagnóstico")
        if abs(calib - 1) >= UMBRALES["calib"]["verde"]:
            st.warning(f"**Calibración fuera de rango.** El modelo predice una PD media de "
                       f"{metrica_de('Scorecard · OOT', 'PD_media'):.2%} y la mora observada fue "
                       f"{metrica_de('Scorecard · OOT', 'Tasa_real'):.2%} (ratio {calib:.2f}). "
                       f"Ordena bien pero subestima el nivel: afecta cualquier uso de la PD para provisiones "
                       f"o precios, no al ranking ni a las bandas. **Acción:** recalibrar el intercepto con la "
                       f"ventana más reciente.")
        peor = D["psi_variables"].PSI.idxmax()
        valor_peor = D["psi_variables"].PSI.max()
        icono, _ = semaforo(valor_peor, "psi")
        st.info(f"**Variable menos estable:** {icono} `{peor}` con PSI {valor_peor:.3f}. "
                f"{'Dentro de rango.' if valor_peor < 0.10 else 'Vigilar la composición de la cartera.'}")
        st.success(f"**Sin sobreajuste.** El gap entre entrenamiento y validación del scorecard es "
                   f"{D['gaps'].query('Modelo == \"Scorecard\"').Gap_AUC.mean():.4f} de AUC, "
                   f"frente a {D['gaps'].query('Modelo == \"LightGBM\"').Gap_AUC.mean():.4f} del challenger.")
    with der:
        st.subheader("Gini por muestra")
        g = pd.Series(D["ficha"]["gini"])
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        g.plot.bar(ax=ax, color=[GRIS, MAGENTA, AMBAR])
        ax.axhline(0.36, ls="--", c="green", lw=.8); ax.axhline(0.30, ls="--", c="red", lw=.8)
        ax.set_ylim(0, max(0.5, g.max() * 1.2)); ax.tick_params(axis="x", rotation=0)
        ax.set_title("Líneas: umbrales de monitoreo")
        st.pyplot(fig, width="stretch")

    st.divider()
    st.subheader("Comparación con el challenger")
    st.dataframe(D["metricas"].set_index("modelo").style.format("{:.4f}"), width="stretch")


def pag_variables():
    st.title("🔍 Explorador de variables del modelo")
    st.caption("Las variables que quedaron en el scorecard final, con todo lo que define su comportamiento.")

    fichas = D["variables"].copy()
    fichas["semáforo PSI"] = [semaforo(x, "psi")[0] for x in fichas.PSI_TRAIN_OOT]
    st.dataframe(
        fichas[["familia", "peso_puntos", "IV", "PSI_TRAIN_OOT", "semáforo PSI", "pct_faltante",
                "coeficiente", "p_valor", "VIF"]]
        .sort_values("peso_puntos", ascending=False)
        .style.format({"peso_puntos": "{:.1f}", "IV": "{:.4f}", "PSI_TRAIN_OOT": "{:.4f}",
                       "pct_faltante": "{:.1%}", "coeficiente": "{:.4f}", "p_valor": "{:.4f}", "VIF": "{:.2f}"}),
        width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        p = fichas.peso_puntos.sort_values()
        colores = [ROSA if f == "Perfil del cliente" else MAGENTA for f in fichas.loc[p.index, "familia"]]
        p.plot.barh(ax=ax, color=colores)
        ax.set_title("Peso de cada variable (rango de puntos)\nrosado = variable de perfil")
        ax.set_ylabel("")
        st.pyplot(fig, width="stretch")
    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        fichas.groupby("familia").peso_puntos.sum().sort_values().plot.barh(ax=ax, color=CIRUELA)
        ax.set_title("Peso agregado por dimensión de riesgo"); ax.set_ylabel("")
        st.pyplot(fig, width="stretch")

    st.divider()
    v = st.selectbox("Variable a inspeccionar", fichas.sort_values("peso_puntos", ascending=False).index)
    f = fichas.loc[v]

    k = st.columns(5)
    k[0].metric("Dimensión", f.familia)
    k[1].metric("Peso", f"{f.peso_puntos:.1f} pts")
    k[2].metric("IV", f"{f.IV:.4f}")
    k[3].metric(f"{semaforo(f.PSI_TRAIN_OOT, 'psi')[0]} PSI", f"{f.PSI_TRAIN_OOT:.4f}")
    k[4].metric("Sin dato", f"{f.pct_faltante:.1%}")

    t = D["puntos"][D["puntos"].variable == v].reset_index(drop=True)
    t["atributo"] = t["atributo"].astype(str).str.replace(r"<ArrowStringArray>\n\[(.*?)\]\n.*", r"\1", regex=True)
    izq, der = st.columns([1.1, 1])
    with izq:
        st.markdown("**Tramos del binning y puntos que otorga cada uno**")
        st.dataframe(t.drop(columns="variable").style.format(
            {"%clientes": "{:.1%}", "tasa_malos": "{:.2%}", "WoE": "{:.4f}", "puntos": "{:.1f}"}),
            width="stretch", hide_index=True)
    with der:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        x = range(len(t))
        ax.bar(x, t.tasa_malos * 100, color=MAGENTA, label="% malos del tramo")
        ax2 = ax.twinx()
        ax2.plot(x, t.puntos, color=AMBAR, marker="o", lw=2, label="puntos")
        ax.set_xticks(list(x)); ax.set_xticklabels(t.atributo.astype(str), rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("% malos"); ax2.set_ylabel("puntos"); ax2.grid(False)
        ax.set_title(f"{v}: riesgo y puntaje por tramo")
        fig.legend(loc="upper right", fontsize=7)
        st.pyplot(fig, width="stretch")
        st.caption("A más riesgo en el tramo, menos puntos. Si la relación se invierte, el binning hay que revisarlo.")

    col = "pts_" + v
    if col in CLI.columns:
        st.markdown("**Cómo se reparten los puntos de esta variable en la cartera**")
        a, b = st.columns([1, 1])
        with a:
            fig, ax = plt.subplots(figsize=(5.2, 2.6))
            for s, c_ in (("TRAIN", GRIS), ("OOT", AMBAR)):
                ax.hist(CLI[CLI.split == s][col], bins=30, alpha=.6, density=True, color=c_, label=s)
            ax.legend(fontsize=8); ax.set_title("Distribución de puntos: desarrollo vs fuera de tiempo")
            st.pyplot(fig, width="stretch")
        with b:
            r = CLI.groupby("split")[col].agg(["mean", "std", "min", "max"]).round(2)
            st.dataframe(r, width="stretch")
            st.caption("Si la media se desplaza entre desarrollo y fuera de tiempo, la variable está cambiando "
                       "de composición: es la señal que anticipa un PSI alto.")


def pag_desempeno():
    st.title("Desempeño")
    muestra = st.radio("Muestra", ["TEST", "OOT"], horizontal=True, index=1)
    d = CLI[CLI.split == muestra]

    k = st.columns(4)
    k[0].metric("Clientes", f"{len(d):,}")
    k[1].metric("Mora observada", f"{d.bad.mean():.2%}")
    k[2].metric("Gini", f"{metrica_de(f'Scorecard · {muestra}', 'Gini'):.4f}")
    k[3].metric("KS", f"{metrica_de(f'Scorecard · {muestra}', 'KS'):.4f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(d.score[d.bad == 0], bins=40, alpha=.6, density=True, color=GRIS, label="Buenos")
        ax.hist(d.score[d.bad == 1], bins=40, alpha=.6, density=True, color=MAGENTA, label="Malos")
        ax.legend(); ax.set_xlabel("Score (mayor = mejor cliente)")
        ax.set_title(f"Distribución del score — {muestra}")
        st.pyplot(fig, width="stretch")
    with c2:
        dd = d.assign(decil=pd.qcut(d.score.rank(method="first"), 10, labels=range(10, 0, -1)))
        t = dd.groupby("decil", observed=True).bad.mean().sort_index()
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        (t * 100).plot.bar(ax=ax, color=MAGENTA)
        ax.axhline(d.bad.mean() * 100, ls="--", c=CIRUELA, lw=1)
        ax.set_title(f"% malos por decil de riesgo — {muestra}\n(1 = más riesgo; línea = promedio)")
        ax.set_xlabel("decil"); ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig, width="stretch")

    st.subheader("Tabla de deciles (fuera de tiempo)")
    st.dataframe(D["deciles"].style.format({"tasa": "{:.2%}", "%malos_acum": "{:.1%}", "lift": "{:.2f}"}),
                 width="stretch")

    st.subheader("Gini por mes")
    g = (CLI[CLI.split != "TRAIN"].groupby("PERIODO")
         .apply(lambda x: 2 * ((x.score.rank().corr(1 - x.bad, method="spearman") + 1) / 2) - 1
                if x.bad.nunique() > 1 else np.nan, include_groups=False))
    st.caption("Aproximación por correlación de rangos; el Gini exacto por muestra está arriba. "
               "Cada mes tiene pocos malos, así que la serie oscila: las decisiones se toman con TEST y OOT agregados.")
    fig, ax = plt.subplots(figsize=(9, 2.8))
    g.plot(ax=ax, marker="o", color=MAGENTA); ax.set_xlabel(""); ax.set_title("Poder discriminante mensual")
    st.pyplot(fig, width="stretch")


def pag_estabilidad():
    st.title("Estabilidad y monitoreo")
    st.caption("El indicador que se revisa mes a mes. Verde < 0.10 · ámbar 0.10–0.25 · rojo > 0.25")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("PSI del score por periodo")
        p = D["psi_mensual"].iloc[:, 0]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        colores = ["green" if x < .10 else AMBAR if x < .25 else "red" for x in p]
        p.plot.bar(ax=ax, color=colores)
        ax.axhline(.10, ls="--", c="green", lw=.8); ax.axhline(.25, ls="--", c="red", lw=.8)
        ax.set_xlabel(""); ax.set_title("Contra la distribución de entrenamiento")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, width="stretch")
    with c2:
        st.subheader("PSI por variable")
        pv = D["psi_variables"].PSI.sort_values(ascending=False)
        tabla = pv.to_frame("PSI")
        tabla["semáforo"] = [semaforo(x, "psi")[0] for x in pv]
        tabla["estado"] = [semaforo(x, "psi")[1] for x in pv]
        st.dataframe(tabla.style.format({"PSI": "{:.4f}"}), width="stretch")

    st.subheader("Mezcla de la cartera por banda de riesgo")
    mezcla = pd.crosstab(CLI.banda, CLI.split, normalize="columns").reindex(BANDAS[::-1]) * 100
    c3, c4 = st.columns([1, 1])
    with c3:
        st.dataframe(mezcla.round(1).add_suffix(" (%)"), width="stretch")
    with c4:
        fig, ax = plt.subplots(figsize=(5.2, 3))
        mezcla.plot.bar(ax=ax, color=[GRIS, ROSA, AMBAR])
        ax.set_title("% de clientes por banda"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")
    st.caption("Si la cartera se corre hacia las bandas de más riesgo entre desarrollo y fuera de tiempo, "
               "parte del deterioro es cambio de composición y no empeoramiento del comportamiento.")


def pag_calibracion():
    st.title("Calibración")
    st.caption("El modelo puede ordenar bien y aun así equivocarse en el nivel. Esto mira el nivel.")

    tabla = (CLI.groupby("split").agg(PD_media=("PD", "mean"), mora_real=("bad", "mean"), n=("bad", "size"))
             .reindex(["TRAIN", "TEST", "OOT"]))
    tabla["obs/pred"] = tabla.mora_real / tabla.PD_media
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(tabla.style.format({"PD_media": "{:.2%}", "mora_real": "{:.2%}",
                                         "obs/pred": "{:.3f}", "n": "{:,.0f}"}), width="stretch")
        ratio = tabla.loc["OOT", "obs/pred"]
        icono, estado = semaforo(abs(ratio - 1), "calib")
        st.metric(f"{icono} obs/pred fuera de tiempo", f"{ratio:.3f}", estado, delta_color="off")
        if ratio > 1.1:
            st.warning("La PD subestima el nivel de mora reciente. **Acción:** recalibrar el intercepto "
                       "con la ventana más reciente. No afecta al ranking ni a las bandas.")
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        x = np.arange(len(tabla))
        ax.bar(x - .2, tabla.PD_media * 100, .4, label="PD media predicha", color=GRIS)
        ax.bar(x + .2, tabla.mora_real * 100, .4, label="Mora observada", color=MAGENTA)
        ax.set_xticks(x); ax.set_xticklabels(tabla.index); ax.legend(fontsize=8)
        ax.set_ylabel("%"); ax.set_title("Predicho contra observado")
        st.pyplot(fig, width="stretch")

    st.subheader("Calibración por banda (fuera de tiempo)")
    o = CLI[CLI.split == "OOT"]
    por_banda = o.groupby("banda", observed=True).agg(n=("bad", "size"), PD_media=("PD", "mean"),
                                                      mora_real=("bad", "mean")).reindex(BANDAS[::-1])
    por_banda["obs/pred"] = por_banda.mora_real / por_banda.PD_media
    st.dataframe(por_banda.style.format({"PD_media": "{:.2%}", "mora_real": "{:.2%}",
                                         "obs/pred": "{:.2f}", "n": "{:,.0f}"}), width="stretch")
    st.caption("Que el total cuadre no garantiza que cada banda cuadre. Esta tabla es la que pediría "
               "un validador independiente.")


def pag_bandas():
    st.title("Bandas y punto de corte")
    o = CLI[CLI.split == "OOT"]

    t = o.groupby("banda", observed=True).agg(clientes=("bad", "size"), mora=("bad", "mean")).reindex(BANDAS[::-1])
    t["% de la cartera"] = t.clientes / t.clientes.sum()
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Bandas de riesgo (fuera de tiempo)")
        st.dataframe(t.style.format({"clientes": "{:,.0f}", "mora": "{:.2%}", "% de la cartera": "{:.1%}"}),
                     width="stretch")
        monotona = t.mora.is_monotonic_decreasing
        if monotona:
            st.success("Las bandas son monótonas: la mora baja de forma consistente de E hacia A.")
        else:
            st.warning("Las bandas no son monótonas en esta muestra: revisar los cortes.")
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        (t.mora * 100).plot.bar(ax=ax, color=[MAGENTA, ROSA, GRIS, "#B9D4C9", "#8FBF9F"][:len(t)])
        ax.set_title("% de mora por banda"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, width="stretch")

    st.divider()
    st.subheader("Simulador de punto de corte")
    st.caption("El corte no lo define el modelo: depende del apetito de riesgo, el margen por cliente "
               "y el costo de la mora. Esto muestra el intercambio.")
    pct = st.slider("% de la cartera que se gestiona (la de mayor riesgo)", 5, 50, 20, step=5)
    corte = o.score.quantile(pct / 100)
    gestionados, resto = o[o.score <= corte], o[o.score > corte]

    k = st.columns(4)
    k[0].metric("Score de corte", f"{corte:.0f}")
    k[1].metric("Clientes gestionados", f"{len(gestionados):,}")
    k[2].metric("Mora capturada", f"{gestionados.bad.sum() / o.bad.sum():.1%}")
    k[3].metric("Mora del resto", f"{resto.bad.mean():.2%}", f"{resto.bad.mean() - o.bad.mean():+.2%} vs cartera",
                delta_color="inverse")

    fig, ax = plt.subplots(figsize=(9, 3))
    q = np.arange(5, 55, 5)
    captura = [o.bad[o.score <= o.score.quantile(x / 100)].sum() / o.bad.sum() * 100 for x in q]
    ax.plot(q, captura, marker="o", color=MAGENTA)
    ax.plot(q, q, "k--", lw=.8, label="gestión al azar")
    ax.axvline(pct, color=AMBAR, lw=2)
    ax.set_xlabel("% de cartera gestionada"); ax.set_ylabel("% de mora capturada")
    ax.set_title("Curva de captura"); ax.legend(fontsize=8)
    st.pyplot(fig, width="stretch")
    st.dataframe(D["cortes"].style.format({"% gestionado": "{:.0%}", "score de corte": "{:.1f}",
                                           "malos capturados": "{:.1%}",
                                           "mora de los no gestionados": "{:.2%}"}),
                 width="stretch", hide_index=True)


def pag_simulador():
    st.title("🧮 Simulador de cliente")
    st.caption("Ingresa un perfil y obtén su score, banda y el aporte de cada variable. "
               "“Sin dato” usa el tramo *Missing* del binning, que es como el modelo trata un faltante estructural.")

    valores, cols = {}, st.columns(3)
    for i, v in enumerate(SEL):
        with cols[i % 3]:
            if v in CATS:
                opciones = NIVELES[v] + ["(sin dato)"]
                moda = MOD.get("moda", {}).get(v, NIVELES[v][0])     # arranca en la categoría más frecuente
                x = st.selectbox(v, opciones, index=opciones.index(moda), key=f"s_{v}")
                valores[v] = np.nan if x == "(sin dato)" else x
            else:
                r = RANGOS[v]
                sin = st.checkbox(f"Sin dato — {v}", key=f"na_{v}",
                                  help=f"{r['pct_faltante']:.0%} de los clientes no tiene este dato")
                valores[v] = np.nan if sin else st.number_input(
                    v, value=float(round(r["mediana"], 2)),
                    min_value=float(min(r["p01"], 0) - abs(r["p01"])),
                    max_value=float(abs(r["p99"]) * 5 + 1), key=f"n_{v}")

    r, pts, W = puntuar(pd.DataFrame([valores]))
    st.divider()
    k = st.columns(4)
    k[0].metric("Score", f"{r.score.iloc[0]:.0f}")
    k[1].metric("PD estimada", f"{r.PD.iloc[0]:.2%}",
                help="Calibrada al periodo de desarrollo. La mora reciente es mayor: ver Calibración.")
    k[2].metric("Banda", str(r.banda.iloc[0]))
    percentil = (CLI.score < r.score.iloc[0]).mean()
    k[3].metric("Percentil en la cartera", f"{percentil:.0%}")

    aporte = (pts.iloc[0] - PUNTOS_BASE).sort_values()
    st.caption("El aporte se mide contra un cliente neutro (WoE = 0 en todas las variables). Un tramo *Missing* "
               "con WoE 0 —como el de una categórica sin dato— aporta exactamente cero por construcción.")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    aporte.plot.barh(ax=ax, color=[MAGENTA if x < 0 else "#2E8F8A" for x in aporte])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title("Aporte de cada variable frente a un cliente promedio (puntos)\n"
                 "verde suma score (menos riesgo) · magenta resta")
    ax.set_ylabel("")
    st.pyplot(fig, width="stretch")

    with st.expander("Detalle: WoE y puntos por variable"):
        det = pd.DataFrame({"WoE": W.iloc[0], "puntos": pts.iloc[0],
                            "aporte vs promedio": aporte.reindex(SEL)})
        st.dataframe(det.style.format("{:.2f}"), width="stretch")
    st.caption("La suma de los puntos de todas las variables es exactamente el score: la tabla de puntos "
               "no es una aproximación del modelo, es el modelo.")


def pag_lote():
    st.title("📂 Scoring por lote y monitoreo de un periodo nuevo")
    st.caption("Sube un CSV con las columnas del modelo. Además de puntuar, compara la distribución del score "
               "contra la de desarrollo: es el control que se corre cada mes.")
    st.code(", ".join(SEL), language=None)

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is None:
        st.info("También sirve el CSV original de la base: se ignoran las columnas que el modelo no usa.")
        return

    try:
        try:
            nuevo = pd.read_csv(archivo, low_memory=False)
        except UnicodeDecodeError:
            archivo.seek(0)
            nuevo = pd.read_csv(archivo, encoding="latin-1", low_memory=False)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return

    faltan = [v for v in SEL if v not in nuevo.columns]
    if faltan:
        st.error(f"Faltan columnas del modelo: {faltan}")
        return

    nuevo = limpiar_como_base(nuevo)
    r, pts, _ = puntuar(nuevo)
    salida = pd.concat([nuevo[[c for c in ("serie", "PERIODO") if c in nuevo.columns]],
                        nuevo[SEL], r[["PD", "score", "banda"]]], axis=1)
    st.success(f"{len(salida):,} clientes puntuados.")

    base_train = CLI[CLI.split == "TRAIN"].score
    psi_lote = psi(base_train, r.score)
    icono, estado = semaforo(psi_lote, "psi")

    k = st.columns(4)
    k[0].metric("Score medio", f"{r.score.mean():.0f}")
    k[1].metric("PD media", f"{r.PD.mean():.2%}")
    k[2].metric(f"{icono} PSI contra desarrollo", f"{psi_lote:.4f}", estado, delta_color="off")
    k[3].metric("% en bandas D y E", f"{r.banda.isin(BANDAS[:2]).mean():.1%}")

    if psi_lote >= 0.25:
        st.error("PSI en rojo: la población de este lote difiere de la de desarrollo. Revisar la fuente de "
                 "datos y considerar recalibrar antes de usar el score para decidir.")
    elif psi_lote >= 0.10:
        st.warning("PSI en ámbar: hay desplazamiento de población. Vigilar en el siguiente corte.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3))
        ax.hist(base_train, bins=40, alpha=.55, density=True, color=GRIS, label="Desarrollo")
        ax.hist(r.score, bins=40, alpha=.55, density=True, color=AMBAR, label="Lote nuevo")
        ax.legend(); ax.set_title("Distribución del score"); ax.set_xlabel("Score")
        st.pyplot(fig, width="stretch")
    with c2:
        comp = pd.DataFrame({
            "Desarrollo": CLI[CLI.split == "TRAIN"].banda.value_counts(normalize=True).reindex(BANDAS[::-1]),
            "Lote nuevo": r.banda.value_counts(normalize=True).reindex(BANDAS[::-1])}) * 100
        fig, ax = plt.subplots(figsize=(5.2, 3))
        comp.plot.bar(ax=ax, color=[GRIS, AMBAR])
        ax.set_title("% de clientes por banda"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, width="stretch")

    st.dataframe(salida.head(100), width="stretch")
    st.download_button("⬇️ Descargar resultados (CSV)", salida.to_csv(index=False).encode(),
                       "scoring_resultados.csv", "text/csv")


def pag_gobierno():
    st.title("📋 Gobierno del modelo")
    f = D["ficha"]

    st.subheader("Ficha del modelo")
    izq, der = st.columns(2)
    with izq:
        st.dataframe(pd.Series({
            "Nombre": f["nombre"], "Versión": str(f["version"]), "Evento": f["evento"],
            "Variables": str(f["n_variables"]), "Semilla": str(f["seed"]),
            "Base total": f"{f['base_total']:,}",
            "Muestra de desarrollo": f"{f['muestra_desarrollo']:,} ({f['sample_frac']:.0%})",
            "Ventana de desarrollo": f["ventana_desarrollo"], "Ventana fuera de tiempo": f["ventana_oot"],
        }, dtype="string").to_frame("valor"), width="stretch")
    with der:
        st.dataframe(pd.Series({
            "Gini TRAIN": f"{f['gini']['TRAIN']:.4f}", "Gini TEST": f"{f['gini']['TEST']:.4f}",
            "Gini OOT": f"{f['gini']['OOT']:.4f}", "KS OOT": f"{f['ks_oot']:.4f}",
            "PSI del score (OOT)": f"{f['psi_score_oot']:.4f}",
            "Calibración obs/pred": f"{f['calibracion_oot_obs_pred']:.3f}",
            "Challenger (LightGBM) Gini OOT": f"{f['challenger_gini_oot']:.4f}",
        }, dtype="string").to_frame("valor"), width="stretch")

    st.subheader("Reglas de negocio: verificación")
    st.caption("Cada regla impuesta se contrastó contra los datos. Dos no sobrevivieron y se eliminaron.")
    st.dataframe(D["reglas"].style.format({"Gini OOT": "{:.4f}", "estabilidad (Jaccard)": "{:.4f}"}),
                 width="stretch")
    st.markdown("""
- **R1 (cobertura ≥ 10%) es la única regla que se mantiene**, por justificación operativa: una variable con dato en
  menos del 10% de los clientes no es monitoreable en producción.
- **R2 (ventana corta por defecto) se descartó**: sobre la base completa cuesta más de dos puntos de Gini y la
  estabilidad que compraba casi desaparece; además eliminaba la variable de una dimensión de riesgo.
- **R3 (tope por familia) se descartó**: peor Gini de todas las variantes. Limitar las variables de perfil es una
  decisión de política, y como tal se declara y se mide, no se esconde en el pipeline.
- **R4 (familias obligatorias) se descartó**: el hueco que iba a tapar lo producía R2.
- **R5 (excluir `INT_SALDO`) se descartó**: la variable es comportamiento del cliente —cuánto interés genera sobre lo
  que debe—, no la tarifa del banco. La circularidad pasa a ser un riesgo monitoreado: su PSI está en el tablero y un
  cambio en la política de precios dispara revalidación.
""")
    with st.expander("Auditoría: por qué se descartó cada variable"):
        aud = D["auditoria"]
        st.dataframe(aud[aud.iloc[:, 0] != "elegible"], width="stretch")

    st.subheader("Decisión del modelo")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.dataframe(D["matriz"].style.format("{:.2f}"), width="stretch")
    with c2:
        st.dataframe(D["sensibilidad"].style.format("{:.2f}"), width="stretch")
    st.info("**La sensibilidad, dicha de frente:** con pesos solo estadísticos ganaría el LightGBM. "
            "La recomendación del scorecard descansa en interpretabilidad, validación independiente y "
            "capacidad de explicar un rechazo, no en superioridad estadística.")

    st.subheader("Variables excluidas por política")
    st.dataframe(D["politica"].style.format("{:.4f}"), width="stretch")
    st.caption(f"Excluidas: {', '.join(f['excluidas_por_politica'])}. "
               "La tabla mide cuánto Gini se resigna por esa decisión.")

    st.subheader("Pendiente antes de producción")
    st.markdown("""
| Pendiente | Acción |
|---|---|
| **Calibración** | recalibrar el intercepto con la ventana más reciente |
| **Equidad** | medir razón de aprobación y obs/pred por grupo; revisar proxies con Cumplimiento |
| **Estabilidad** | monitoreo mensual con los semáforos de este aplicativo |
| **Gobierno** | categorización por materialidad, inventario de modelos, validación independiente y back-testing (Res. SBS 53-2023) |

**Limitaciones del caso:** el horizonte del evento no está documentado en la bitácora; hay solo 8 meses de
historia y el fuera de tiempo son 2 meses; la base no incluye ingresos, comportamiento interno ni clientes
rechazados, y el evento resultó casi independiente del atraso previo.
""")


PAGINAS = {
    "Salud del modelo": pag_salud,
    "Explorador de variables": pag_variables,
    "Desempeño": pag_desempeno,
    "Estabilidad y monitoreo": pag_estabilidad,
    "Calibración": pag_calibracion,
    "Bandas y punto de corte": pag_bandas,
    "Simulador de cliente": pag_simulador,
    "Scoring por lote": pag_lote,
    "Gobierno del modelo": pag_gobierno,
}


def main():
    st.sidebar.title("Gestión del modelo")
    st.sidebar.caption("Scorecard *early* de morosidad")
    pagina = st.sidebar.radio("Sección", list(PAGINAS))
    st.sidebar.divider()
    f = D["ficha"]
    icono, _ = semaforo(f["gini"]["OOT"], "gini")
    st.sidebar.markdown(f"""**Estado general** {icono}
Gini OOT `{f['gini']['OOT']:.3f}` · KS `{f['ks_oot']:.3f}`
PSI `{f['psi_score_oot']:.3f}` · obs/pred `{f['calibracion_oot_obs_pred']:.2f}`

**Modelo** {f['n_variables']} variables
**Escala** {ESCALA['score_base']} pts = odds {ESCALA['odds_base']}:1, PDO {ESCALA['PDO']}
**Muestra** {f['muestra_desarrollo']:,} ({f['sample_frac']:.0%} de la base)""")
    PAGINAS[pagina]()


if __name__ == "__main__" or True:
    import os
    if os.environ.get("APP_SMOKE"):
        for nombre, fn in PAGINAS.items():
            if nombre == "Scoring por lote":
                continue
            print(">>", nombre)
            fn()
        base = pd.read_csv("/mnt/user-data/uploads/Prueba_Modelos_Caso_Morosidad.csv",
                           encoding="latin-1", low_memory=False).head(500)
        r, _, _ = puntuar(limpiar_como_base(base))
        print(r[["PD", "score", "banda"]].describe().to_string())
        print("PSI lote:", round(psi(CLI[CLI.split == "TRAIN"].score, r.score), 4))
        print("SMOKE_OK")
    else:
        main()