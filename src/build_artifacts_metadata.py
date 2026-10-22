"""Genera artifacts/metadata.json y artifacts/model_manifest.csv.

Fuente: la revisión final ya calculada (dashboard_data/final_review_20261018/...).
No entrena nada. El manifiesto lista los 512 pipelines finales con su semilla,
hiperparámetros elegidos, MSE interno y SHA-256 de la serialización original.

Uso:  python src/build_artifacts_metadata.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R = (REPO / "dashboard_data" / "final_review_20261018" / "files" / "results"
     / "final_regression_20261018")
OUT = REPO / "artifacts"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    proto = json.loads((R / "protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((R / "selection_freeze.json").read_text(encoding="utf-8"))
    sel = pd.read_csv(R / "selected_inputs.csv")
    ms = pd.read_csv(R / "outer_fold_mean_sd.csv")

    manifest = sel[["model", "family", "method", "symbol", "volatility_window", "seed",
                    "params", "inner_mse", "n_train", "train_target_end",
                    "model_path", "model_sha256"]].copy()
    manifest.to_csv(OUT / "model_manifest.csv", index=False)

    meta = {
        "descripcion": "Metadatos de los 512 pipelines finales (revisión del 18-oct-2026).",
        "semilla_global": proto["estimator_seed"],
        "semilla_busqueda": proto["search_seed"],
        "librerias": proto["versions"],
        "objetivo": proto["target"],
        "esquema_entrada": {
            "activos": proto["assets"],
            "ventanas_volatilidad": proto["target_windows"],
            "horizontes": proto["horizons"],
            "salidas": len(proto["horizons"]),
            "entrenamiento_final": "660 orígenes (2023-2024), etiquetas anteriores a 2025",
            "nota_features": proto["har_features"],
        },
        "protocolo": {"externo": proto["outer"], "interno": proto["inner"],
                      "purga": proto["purge"], "objetivo_interno": proto["primary_objective"],
                      "presupuesto_por_busqueda": proto["budget_per_search"]},
        "modelos_finales": int(len(manifest)),
        "metricas_bucle_externo": ms.round(6).to_dict(orient="records"),
        "hash_protocolo": freeze["protocol_sha256"],
        "hash_selected_inputs": freeze["selected_inputs_sha256"],
        "nota": ("Los .joblib de los 512 pipelines no se versionan en este repositorio (peso); "
                 "model_manifest.csv conserva el SHA-256 original de cada uno. La prueba de ida y "
                 "vuelta se ejecuta sobre los modelos serializados que sí están versionados "
                 "(results/optimized_minute_ridge_2023_2025/models)."),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(manifest)} modelos -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
