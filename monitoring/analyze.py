"""
Monitoring dashboard.

Levanta los logs de predicciones y arma 6 paneles de monitoreo
sin depender de ground truth (que para predicciones futuras no
existe).

Run:
    streamlit run monitoring/analyze.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

LOGS_PATH = Path(__file__).parent / "logs" / "predictions.jsonl"
HISTORICAL_CSV = Path(__file__).parents[1] / "model-building" / "food_price_inflation.csv"
HISTORICAL_YEARS = 10

ANOMALY_Z_THRESHOLD = 3.0
OOD_FUTURE_MONTHS_WARN = 24
OOD_FUTURE_MONTHS_HEAVY = 60


@st.cache_data(ttl=300)
def load_historical() -> pd.DataFrame:
    if not HISTORICAL_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORICAL_CSV, usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    df.columns = ["country", "date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=10)
def load_logs() -> pd.DataFrame:
    if not LOGS_PATH.exists() or LOGS_PATH.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_json(LOGS_PATH, lines=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target_date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def ood_score(row: pd.Series) -> int:
    score = 0
    if not row.get("country_in_training", True):
        score += 2
    months = row.get("months_from_last_known")
    if months is not None and months > 0:
        if months > OOD_FUTURE_MONTHS_HEAVY:
            score += 2
        elif months > OOD_FUTURE_MONTHS_WARN:
            score += 1
    return score


def ood_label(score: int) -> str:
    if score >= 3:
        return "🔴 Alto"
    if score >= 1:
        return "🟡 Medio"
    return "🟢 Bajo"


def panel_usage(df: pd.DataFrame) -> None:
    st.subheader("A — Usage analytics")
    st.caption("Quién usa el modelo y cómo")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicciones totales", len(df))
    c2.metric("Países distintos", df["country"].nunique())
    c3.metric("Versiones de modelo", df["model_version"].nunique())
    c4.metric("Fecha futura (predicciones)", int(df["is_future_prediction"].sum()))

    left, right = st.columns(2)
    with left:
        top_countries = df["country"].value_counts().head(10).reset_index()
        top_countries.columns = ["country", "count"]
        st.plotly_chart(
            px.bar(top_countries, x="country", y="count", title="Top 10 países más predichos"),
            use_container_width=True,
        )
    with right:
        per_year = df.groupby(df["target_date"].dt.year).size().reset_index(name="count")
        per_year.columns = ["year", "count"]
        st.plotly_chart(
            px.bar(per_year, x="year", y="count", title="Predicciones por año target"),
            use_container_width=True,
        )

    requests_per_day = df.groupby(df["timestamp"].dt.date).size().reset_index(name="count")
    requests_per_day.columns = ["date", "count"]
    st.plotly_chart(
        px.line(requests_per_day, x="date", y="count", title="Volumen diario de requests"),
        use_container_width=True,
    )


def panel_distribution(df: pd.DataFrame) -> None:
    st.subheader("B — Distribución de predicciones")
    st.caption("¿Las predicciones caen en un rango sano? Auto-calibrado desde los logs (p5/p95).")

    preds = df["prediction"].dropna()
    if preds.empty:
        st.info("Sin predicciones para analizar todavía.")
        return

    p5, p95 = preds.quantile([0.05, 0.95])
    c1, c2, c3 = st.columns(3)
    c1.metric("Mediana", f"{preds.median():.2f}%")
    c2.metric("p5 — p95", f"{p5:.2f}% — {p95:.2f}%")
    c3.metric("Fuera de p5–p95", int(((preds < p5) | (preds > p95)).sum()))

    fig = px.histogram(preds, nbins=40, title="Histograma de predicciones loggeadas")
    fig.add_vline(x=p5, line_dash="dash", annotation_text="p5")
    fig.add_vline(x=p95, line_dash="dash", annotation_text="p95")
    st.plotly_chart(fig, use_container_width=True)


def panel_anomaly(df: pd.DataFrame) -> None:
    st.subheader("C — Anomaly detection")
    st.caption("Predicciones que se alejan mucho del comportamiento del modelo para ese país.")

    work = df.dropna(subset=["prediction"]).copy()
    if work.empty:
        st.info("Sin predicciones para analizar.")
        return

    grp = work.groupby("country")["prediction"]
    work["country_mean"] = grp.transform("mean")
    work["country_std"] = grp.transform("std").fillna(0)
    work["zscore"] = (work["prediction"] - work["country_mean"]) / work["country_std"].replace(0, pd.NA)
    work["zscore"] = work["zscore"].fillna(0)

    anomalies = work[work["zscore"].abs() > ANOMALY_Z_THRESHOLD]
    st.metric(f"Anomalías (|z| > {ANOMALY_Z_THRESHOLD})", len(anomalies))

    if anomalies.empty:
        st.success("Sin anomalías detectadas en los logs actuales.")
    else:
        st.dataframe(
            anomalies[["timestamp", "country", "year", "month", "prediction", "zscore"]]
            .sort_values("zscore", key=abs, ascending=False)
            .head(20),
            use_container_width=True,
        )


def panel_ood(df: pd.DataFrame) -> None:
    st.subheader("D — Out-of-distribution scoring")
    st.caption("¿Cuán fuera del soporte del training está cada query? (país conocido + extrapolación temporal)")

    work = df.copy()
    work["ood_score"] = work.apply(ood_score, axis=1)
    work["ood_level"] = work["ood_score"].apply(ood_label)

    counts = work["ood_level"].value_counts().reset_index()
    counts.columns = ["level", "count"]
    st.plotly_chart(
        px.bar(counts, x="level", y="count", title="Distribución de OOD score", color="level",
               color_discrete_map={"🟢 Bajo": "#2ecc71", "🟡 Medio": "#f1c40f", "🔴 Alto": "#e74c3c"}),
        use_container_width=True,
    )

    high_ood = work[work["ood_score"] >= 3]
    if not high_ood.empty:
        st.warning(f"{len(high_ood)} predicciones con OOD alto — se extrapolan más de 5 años o el país no estaba en training.")
        st.dataframe(
            high_ood[["timestamp", "country", "year", "month", "months_from_last_known", "country_in_training", "prediction"]]
            .head(20),
            use_container_width=True,
        )


def panel_vs_naive(df: pd.DataFrame) -> None:
    st.subheader("F — Modelo vs naive baseline")
    st.caption("Cuánto se aleja el modelo de simplemente repetir el último valor conocido del país.")

    work = df.dropna(subset=["prediction", "last_known_value"]).copy()
    if work.empty:
        st.info("Sin datos suficientes (logs sin last_known_value).")
        return

    work["delta"] = work["prediction"] - work["last_known_value"]
    work["abs_delta"] = work["delta"].abs()

    c1, c2 = st.columns(2)
    c1.metric("Delta absoluto promedio", f"{work['abs_delta'].mean():.2f}")
    c2.metric("Predicciones que coinciden ±0.5", int((work["abs_delta"] <= 0.5).sum()))

    fig = px.scatter(
        work,
        x="last_known_value",
        y="prediction",
        color="months_from_last_known",
        hover_data=["country", "year", "month"],
        title="Modelo vs naive (lag1) — diagonal = idénticos",
    )
    lo = min(work["last_known_value"].min(), work["prediction"].min())
    hi = max(work["last_known_value"].max(), work["prediction"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)


def panel_top_country_vs_history(df: pd.DataFrame, hist: pd.DataFrame) -> None:
    st.subheader("G — País más predicho: histórico vs predicciones")
    st.caption(
        f"Para el país más consultado, dibujamos los últimos {HISTORICAL_YEARS} años de inflación real "
        "(serie histórica) y por encima las predicciones futuras loggeadas. Sirve como sanity check visual: "
        "¿las predicciones se ven como una continuación natural de la serie, o se despegan?"
    )

    if df.empty:
        st.info("Sin logs.")
        return
    if hist.empty:
        st.warning("No se encontró el CSV histórico.")
        return

    top_country = df["country"].value_counts().idxmax()
    top_count = int(df["country"].value_counts().max())
    st.metric(f"Más predicho", f"{top_country} ({top_count} predicciones)")

    country_hist = hist[hist["country"] == top_country].sort_values("date")
    if country_hist.empty:
        st.warning(f"{top_country} no aparece en el CSV histórico — probablemente OOD.")
        return

    cutoff = country_hist["date"].max() - pd.DateOffset(years=HISTORICAL_YEARS)
    country_hist = country_hist[country_hist["date"] >= cutoff]

    country_preds = df[df["country"] == top_country].dropna(subset=["prediction"])

    fig = px.line(
        country_hist,
        x="date",
        y="value",
        title=f"{top_country} — últimos {HISTORICAL_YEARS} años (real) + predicciones loggeadas",
        labels={"date": "Fecha", "value": "Inflación alimentaria (%)"},
    )
    fig.data[0].name = "Histórico real"
    fig.data[0].showlegend = True

    if not country_preds.empty:
        fig.add_scatter(
            x=country_preds["target_date"],
            y=country_preds["prediction"],
            mode="markers",
            name="Predicciones",
            marker=dict(size=9, color="#e74c3c", symbol="diamond"),
            hovertext=[
                f"{r.year}-{r.month:02d} → {r.prediction:.2f}%"
                for r in country_preds.itertuples()
            ],
            hoverinfo="text",
        )

    last_real_date = country_hist["date"].max().to_pydatetime()
    fig.add_vline(
        x=last_real_date.timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="último dato real",
        annotation_position="top",
    )

    st.plotly_chart(fig, use_container_width=True)


def panel_drilldown(df: pd.DataFrame) -> None:
    st.subheader("E — Drill-down por predicción")
    st.caption("Inspección puntual: para una predicción específica, todos los flags y comparaciones.")

    if df.empty:
        st.info("Sin logs.")
        return

    options = (
        df.assign(label=df["country"] + " — " + df["target_date"].dt.strftime("%Y-%m"))
        .drop_duplicates(subset=["label"])
        .sort_values("timestamp", ascending=False)
    )
    selected_label = st.selectbox("Seleccionar predicción", options["label"].tolist())
    row = options[options["label"] == selected_label].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicción", f"{row['prediction']:.2f}%" if pd.notna(row["prediction"]) else "—")
    c2.metric("Naive (lag1)", f"{row['last_known_value']:.2f}%" if pd.notna(row.get("last_known_value")) else "—")
    delta_naive = (
        row["prediction"] - row["last_known_value"]
        if pd.notna(row["prediction"]) and pd.notna(row.get("last_known_value"))
        else None
    )
    c3.metric("Δ vs naive", f"{delta_naive:+.2f}" if delta_naive is not None else "—")

    score = ood_score(row)
    flags = {
        "País en training": "✅" if row.get("country_in_training") else "❌",
        "Predicción a futuro": "🔮 sí" if row.get("is_future_prediction") else "📚 histórica",
        "Meses desde último dato conocido": row.get("months_from_last_known"),
        "Última fecha conocida del país": row.get("last_known_date"),
        "OOD score": f"{score} ({ood_label(score)})",
    }
    st.json(flags)

    same_country = df[df["country"] == row["country"]].sort_values("target_date")
    if len(same_country) > 1:
        fig = px.line(
            same_country,
            x="target_date",
            y="prediction",
            title=f"Predicciones loggeadas para {row['country']}",
            markers=True,
        )
        fig.add_scatter(
            x=[row["target_date"]],
            y=[row["prediction"]],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="Esta predicción",
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Monitoring — CD4ML", layout="wide")
    st.title("Monitoring Dashboard — Inflación Alimentaria")
    st.caption("Cierra el feedback loop de CD4ML. Todo deriva de monitoring/logs/predictions.jsonl.")

    df = load_logs()
    hist = load_historical()
    if df.empty:
        st.warning("No hay logs todavía. Levantá la API y hacé alguna predicción para popular el dashboard.")
        return

    st.divider()
    panel_usage(df)
    st.divider()
    panel_distribution(df)
    st.divider()
    panel_anomaly(df)
    st.divider()
    panel_ood(df)
    st.divider()
    panel_vs_naive(df)
    st.divider()
    panel_top_country_vs_history(df, hist)
    st.divider()
    panel_drilldown(df)


if __name__ == "__main__":
    main()
