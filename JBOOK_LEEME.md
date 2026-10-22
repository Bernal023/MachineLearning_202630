# Jupyter Book del informe técnico final

Añadir estos archivos a la raíz del repositorio (respetando las rutas):

```
book/_config.yml  book/_toc.yml  book/references.bib  book/jbook_utils.py
book/intro.md  book/cap01_… cap09_….md  book/referencias.md
artifacts/{metadata.json, model_manifest.csv}
src/build_artifacts_metadata.py  src/check_jbook_cells.py
requirements-book.txt   .github/workflows/jupyter-book-pages.yml
```

1. **Desactivar el despliegue anterior:** borrar `.github/workflows/deploy.yml` (MyST/`mystmd`); ambos publican en Pages y se pisarían.
2. En GitHub: *Settings → Pages → Source: GitHub Actions*.
3. `git add . && git commit -m "JBook informe final" && git push` → el workflow compila y publica en
   https://jassan-bit.github.io/MachineLearning_202630/
4. Local: `pip install -r requirements-book.txt && jupyter-book build book/`.
   Verificación rápida sin Sphinx: `python src/check_jbook_cells.py`.

`book/sections/`, `book/figures/` y `entregable1_master.md` (sitio MyST anterior) quedan excluidos del libro vía `exclude_patterns`.
El libro lee `dashboard_data/…`, `data/processed/daily_2020_2025.csv` y `results/optimized_minute_ridge_2023_2025/models/`; deben seguir versionados.
