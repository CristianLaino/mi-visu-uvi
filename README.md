# Visor UVI — Streamlit

Aplicación interactiva para visualizar climatologías y observaciones de índice UV (UVI) a partir de los datos procesados en `prueba.py`.

La app permite:
- Seleccionar la climatología (media suavizada, N_smooth = 30) y la variable de referencia.
- Mostrar observaciones (overpass) o series diarias agregadas (media, máximo, mínimo).
- Elegir el tipo de marcador / línea y estilo de visualización (estándar o académico).
- Exportar la figura como PNG/HTML y descargar los datos visibles en CSV.

---

## Estructura del repositorio

mi-visu-uvi/
├─ app_streamlit.py # App Streamlit (interfaz principal)
├─ prueba.py # Script que prepara df, df_daily_uv, vars_uv, etc.
├─ requirements.txt # Dependencias
├─ README.md
├─ .gitignore
├─ data/ # <opcional> datos (no subir datos privados)
└─ assets/ # <opcional> logos, imágenes

---

## Requisitos

Python 3.8+ recomendado.

Dependencias (ejemplo en `requirements.txt`):
streamlit
pandas
numpy
plotly
kaleido # opcional, para exportar PNG


Instalar:
```bash
pip install -r requirements.txt
