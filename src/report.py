"""
report.py
=========
Genera el reporte HTML final interactivo (results/reporte_final.html) a partir
de results/all_results.csv y los artefactos por experimento.

Contenido:
  - Resumen ejecutivo + leaderboard (mejores modelos)
  - Comparacion de modelos (contribucion iii)
  - Efecto del rango / fusion VIS-NIR (contribucion i)
  - Efecto del preprocesado espectral (contribucion i)
  - Dispersion medido vs predicho del mejor modelo
  - Curvas de entrenamiento del mejor modelo
  - Importancia de longitudes de onda (contribucion ii)
  - Bloque JSON legible por maquina al final (para diagnostico/iteracion)

El HTML es autocontenido (Plotly via CDN). Puedes abrirlo en el navegador o
enviarmelo para que interprete los resultados y proponga mejoras.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

import config


def _fig(fig, first=False):
    return to_html(fig, full_html=False,
                   include_plotlyjs="cdn" if first else False,
                   default_height=420)


def _agg(df):
    """Agrega por (model, range, preprocess, aggregate) promediando sobre semillas."""
    df = df[df.status == "ok"].copy()
    if "aggregate" not in df.columns:
        df["aggregate"] = "median"
    df["aggregate"] = df["aggregate"].fillna("median")
    keys = ["model", "range", "preprocess", "aggregate"]
    g = (df.groupby(keys)
         .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
              rmse_mean=("rmse", "mean"), rpd_mean=("rpd", "mean"),
              n=("r2", "size"))
         .reset_index()
         .sort_values("r2_mean", ascending=False))
    return g


def generate_report(results_csv=None, out_html=None, logger=None):
    results_csv = Path(results_csv or (config.RESULTS_DIR / "all_results.csv"))
    out_html = Path(out_html or (config.RESULTS_DIR / "reporte_final.html"))
    df = pd.read_csv(results_csv)
    # Fusiona baselines clasicos (PLSR/RF/SVR) si existen en archivo separado.
    classical = config.RESULTS_DIR / "classical_results.csv"
    if classical.exists():
        df = pd.concat([df, pd.read_csv(classical)], ignore_index=True)
    ok = df[df.status == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No hay experimentos exitosos en " + str(results_csv))

    if "aggregate" not in ok.columns:
        ok["aggregate"] = "median"
    ok["aggregate"] = ok["aggregate"].fillna("median")
    agg = _agg(df)
    best = agg.iloc[0]
    best_exp = (ok[(ok.model == best.model) & (ok["range"] == best["range"]) &
                   (ok.preprocess == best.preprocess) &
                   (ok["aggregate"] == best["aggregate"])]
                .sort_values("r2", ascending=False).iloc[0])

    blocks = []

    # ---- 1. Comparacion de modelos (mejor config por modelo) ----
    by_model = agg.sort_values("r2_mean", ascending=False).groupby("model").first().reset_index()
    f = px.bar(by_model.sort_values("r2_mean"), x="r2_mean", y="model",
               orientation="h", error_x="r2_std", color="r2_mean",
               color_continuous_scale="Viridis",
               title="Comparacion de modelos (mejor configuracion de cada uno) - R2",
               labels={"r2_mean": "R2 (test)", "model": "Modelo"})
    f.add_vline(x=config.TARGET_R2, line_dash="dash", line_color="red",
                annotation_text=f"Objetivo R2={config.TARGET_R2}")
    blocks.append(("Contribucion iii: potencial de los modelos", _fig(f, first=True)))

    # ---- 2. Efecto del rango / fusion VIS-NIR ----
    by_range = (ok.groupby(["range", "model"]).r2.mean().reset_index())
    f = px.bar(by_range, x="range", y="r2", color="model", barmode="group",
               category_orders={"range": ["VIS", "NIR", "VISNIR"]},
               title="Efecto del rango espectral y de la fusion VIS-NIR (R2 medio)",
               labels={"r2": "R2 (test)", "range": "Rango"})
    blocks.append(("Contribucion i: efecto de la fusion VIS-NIR", _fig(f)))

    # ---- 3. Efecto del preprocesado ----
    by_prep = ok.groupby(["preprocess", "model"]).r2.mean().reset_index()
    f = px.bar(by_prep, x="preprocess", y="r2", color="model", barmode="group",
               title="Efecto del preprocesado espectral (R2 medio)",
               labels={"r2": "R2 (test)", "preprocess": "Preprocesado"})
    blocks.append(("Contribucion i: efecto del preprocesado", _fig(f)))

    # ---- 4. Dispersion medido vs predicho (mejor modelo) ----
    best_dir = config.RESULTS_DIR / best_exp["exp_name"]
    pred_csv = best_dir / "predictions.csv"
    if pred_csv.exists():
        p = pd.read_csv(pred_csv)
        lim = [min(p.y_true.min(), p.y_pred.min()), max(p.y_true.max(), p.y_pred.max())]
        f = go.Figure()
        f.add_trace(go.Scatter(x=p.y_true, y=p.y_pred, mode="markers",
                               marker=dict(color="#2c7fb8", opacity=0.7),
                               name="muestras"))
        f.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                               line=dict(dash="dash", color="black"), name="1:1"))
        f.update_layout(title=f"Medido vs Predicho - Mejor modelo "
                              f"({best.model}, {best['range']}, {best.preprocess}) "
                              f"R2={best_exp.r2:.3f} RMSE={best_exp.rmse:.3f}%",
                        xaxis_title="COS medido (%)", yaxis_title="COS predicho (%)")
        blocks.append(("Validacion del mejor modelo (vs laboratorio)", _fig(f)))

    # ---- 5. Curvas de entrenamiento (mejor modelo) ----
    hist_csv = best_dir / "history.csv"
    if hist_csv.exists():
        h = pd.read_csv(hist_csv)
        f = go.Figure()
        if "loss" in h: f.add_trace(go.Scatter(y=h["loss"], name="train loss"))
        if "val_loss" in h: f.add_trace(go.Scatter(y=h["val_loss"], name="val loss"))
        f.update_layout(title="Curvas de entrenamiento (mejor modelo)",
                        xaxis_title="epoca", yaxis_title="MSE (escalado)")
        blocks.append(("Curvas de entrenamiento", _fig(f)))

    # ---- 6. Importancia de longitudes de onda ----
    imp_npz = best_dir / "wavelength_importance.npz"
    if imp_npz.exists():
        d = np.load(imp_npz, allow_pickle=True)
        wl, imp = d["wavelengths"], d["importance"]
        f = go.Figure()
        f.add_trace(go.Scatter(x=wl, y=imp, mode="lines", line=dict(color="#d95f02")))
        top = d["top"].tolist() if "top" in d else []
        if top:
            f.add_trace(go.Scatter(x=[t[0] for t in top], y=[t[1] for t in top],
                                   mode="markers+text", marker=dict(color="red", size=9),
                                   text=[f"{t[0]:.0f}" for t in top],
                                   textposition="top center", name="top bandas"))
        f.update_layout(title="Importancia de longitudes de onda para COS",
                        xaxis_title="Longitud de onda (nm)", yaxis_title="Importancia (norm.)")
        blocks.append(("Contribucion ii: longitudes de onda relevantes", _fig(f)))

    # ---- Tabla resumen ----
    table = agg.head(20).round(3).to_html(index=False, classes="tbl", border=0)

    # ---- JSON legible por maquina ----
    summary = {
        "n_experiments_ok": int(len(ok)),
        "target_r2": config.TARGET_R2,
        "best": {
            "model": best.model, "range": best["range"],
            "preprocess": best.preprocess,
            "r2_mean": round(float(best.r2_mean), 4),
            "rmse_mean": round(float(best.rmse_mean), 4),
            "rpd_mean": round(float(best.rpd_mean), 3),
        },
        "by_model_best": by_model.round(4).to_dict(orient="records"),
        "by_range_mean": ok.groupby("range").r2.mean().round(4).to_dict(),
        "by_preprocess_mean": ok.groupby("preprocess").r2.mean().round(4).to_dict(),
        "top10_configs": agg.head(10).round(4).to_dict(orient="records"),
    }

    achieved = "SI" if best.r2_mean >= config.TARGET_R2 else "AUN NO"
    achieved_cls = "ok" if best.r2_mean >= config.TARGET_R2 else "no"
    html = _TEMPLATE.format(
        achieved=achieved, achieved_cls=achieved_cls,
        best_model=best.model, best_range=best["range"], best_prep=best.preprocess,
        best_r2=f"{best.r2_mean:.3f}", best_rmse=f"{best.rmse_mean:.3f}",
        best_rpd=f"{best.rpd_mean:.2f}", target_r2=config.TARGET_R2,
        n_ok=len(ok), n_total=len(df),
        sections="\n".join(
            f'<section><h2>{t}</h2>{c}</section>' for t, c in blocks),
        table=table,
        json_summary=json.dumps(summary, indent=2, ensure_ascii=False),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    if logger:
        logger.info("Reporte HTML generado: %s", out_html)
    return out_html


_TEMPLATE = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Reporte COS - Deep Learning</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}}
 header{{background:#1b4965;color:#fff;padding:28px 40px}}
 header h1{{margin:0 0 6px}}
 .kpi{{display:flex;gap:18px;flex-wrap:wrap;padding:24px 40px}}
 .card{{background:#fff;border-radius:10px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.1);flex:1;min-width:160px}}
 .card .v{{font-size:26px;font-weight:700;color:#1b4965}}
 .card .l{{font-size:13px;color:#666}}
 section{{background:#fff;margin:18px 40px;padding:18px 24px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 section h2{{color:#1b4965;border-bottom:2px solid #eee;padding-bottom:8px}}
 .tbl{{border-collapse:collapse;width:100%;font-size:13px}}
 .tbl th,.tbl td{{border-bottom:1px solid #eee;padding:6px 10px;text-align:left}}
 .tbl th{{background:#1b4965;color:#fff}}
 pre{{background:#0f1b22;color:#9fe;padding:16px;border-radius:8px;overflow:auto;font-size:12px}}
 .ok{{color:#2e7d32;font-weight:700}} .no{{color:#c62828;font-weight:700}}
</style></head><body>
<header>
 <h1>Estimacion de Carbono Organico del Suelo &mdash; Reporte de Deep Learning</h1>
 <div>Comparacion de modelos, fusion VIS-NIR y seleccion de longitudes de onda</div>
</header>
<div class="kpi">
 <div class="card"><div class="v">{best_model}</div><div class="l">Mejor modelo</div></div>
 <div class="card"><div class="v">{best_r2}</div><div class="l">R2 (objetivo {target_r2})</div></div>
 <div class="card"><div class="v">{best_rmse}%</div><div class="l">RMSE</div></div>
 <div class="card"><div class="v">{best_rpd}</div><div class="l">RPD</div></div>
 <div class="card"><div class="v">{best_range}</div><div class="l">Rango</div></div>
 <div class="card"><div class="v">{best_prep}</div><div class="l">Preprocesado</div></div>
 <div class="card"><div class="v">{n_ok}/{n_total}</div><div class="l">Experimentos OK</div></div>
</div>
<section><h2>Resumen ejecutivo</h2>
 <p>Se ejecutaron <b>{n_ok}</b> experimentos exitosos. La mejor combinacion fue
 <b>{best_model}</b> con rango <b>{best_range}</b> y preprocesado <b>{best_prep}</b>,
 alcanzando <b>R2={best_r2}</b> y <b>RMSE={best_rmse}%</b>.
 Objetivo R2&ge;{target_r2}: <span class="{achieved_cls}">{achieved}</span>.</p>
</section>
{sections}
<section><h2>Tabla resumen (top 20 configuraciones)</h2>{table}</section>
<section><h2>Resumen legible por maquina (JSON)</h2>
 <p>Copia este bloque (o envia el HTML) para diagnostico e iteracion de mejoras.</p>
 <pre>{json_summary}</pre></section>
</body></html>"""
