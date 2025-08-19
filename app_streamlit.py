# app_streamlit.py
"""
Visor UVI — visualización afinada
- Selección tipo de puntos: semáforo / punto sin color / línea / línea+punto
- Selección agregado: cada valor / promedio diario / máximo diario / mínimo diario
- Opción para guardar imagen (PNG via kaleido) y HTML interactivo
- Estilos: 'actual' o 'académico'
- Climatología: reconstruida filtrando percentiles 1-99 y suavizada con N_smooth (30)
- Requiere: streamlit, pandas, numpy, plotly, kaleido
"""

import importlib
from datetime import timedelta
import io

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

MODULE_NAME = "prueba"

# -------------------------
# Cargar / recargar prueba.py (usar cache_resource para módulos)
# -------------------------
@st.cache_resource
def load_prueba_module():
    mod = importlib.import_module(MODULE_NAME)
    importlib.reload(mod)
    return mod

def force_reload_prueba():
    try:
        load_prueba_module.clear()
    except Exception:
        pass
    mod = importlib.import_module(MODULE_NAME)
    importlib.reload(mod)
    return mod

# -------------------------
# Reconstruir climatología desde df_daily_uv con filtro percentiles 1-99 y suavizado
# -------------------------
@st.cache_data(show_spinner=False)
def build_climatology_from_daily(df_daily_uv_in, vars_uv_list, N_smooth=30):
    df_daily = df_daily_uv_in.copy()
    df_daily = df_daily.dropna(how='all')
    for var in vars_uv_list:
        if var not in df_daily.columns:
            continue
        p1, p99 = df_daily[var].quantile([0.01, 0.99])
        df_daily = df_daily[df_daily[var].between(p1, p99)]
    if 'DOY' not in df_daily.columns:
        if isinstance(df_daily.index, pd.DatetimeIndex):
            df_daily = df_daily.copy()
            df_daily['DOY'] = df_daily.index.dayofyear
        else:
            raise ValueError("df_daily_uv no tiene columna 'DOY' y no es DatetimeIndex.")
    df_daily['DOY'] = df_daily['DOY'].astype(int)
    df_daily.loc[df_daily['DOY'] == 366, 'DOY'] = 365
    clim = df_daily.groupby('DOY')[vars_uv_list].agg(['mean', 'std', 'count'])
    clim.columns = [f"{v}_{stat}" for v, stat in clim.columns]
    clim = clim.sort_index()
    for v in vars_uv_list:
        mcol = f"{v}_mean"; scol = f"{v}_std"
        ms = f"{v}_mean_smooth"; ss = f"{v}_std_smooth"
        if mcol in clim.columns:
            clim[ms] = clim[mcol].rolling(window=N_smooth, center=True, min_periods=1).mean()
        else:
            clim[ms] = np.nan
        if scol in clim.columns:
            clim[ss] = clim[scol].rolling(window=N_smooth, center=True, min_periods=1).mean()
        else:
            clim[ss] = np.nan
    return clim

# -------------------------
# Iniciar app
# -------------------------
st.set_page_config(page_title="Visor UVI — Afinado", layout="wide")
st.title("Visor UVI — visualización afinada")

# Botón recarga
reload_col, info_col = st.columns([1, 4])
with reload_col:
    if st.button("🔄 Recargar prueba.py (force)"):
        try:
            prueba = force_reload_prueba()
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error recargando: {e}")

with info_col:
    st.markdown("Usa los controles para definir la visualización: estilo, tipo de puntos, agregado, y exportación.")

# cargar módulo prueba.py
try:
    prueba = load_prueba_module()
except Exception as e:
    st.error(f"No se pudo importar '{MODULE_NAME}': {e}")
    st.stop()

# verificar variables obligatorias
required = ['df', 'df_daily_uv', 'vars_uv']
missing = [r for r in required if not hasattr(prueba, r)]
if missing:
    st.error(f"prueba.py debe contener las variables: {missing}")
    st.stop()

# obtener objetos
df = prueba.df.copy()
df_daily_uv = prueba.df_daily_uv.copy()
vars_uv = list(prueba.vars_uv)
N_smooth = getattr(prueba, 'N_smooth', 30)
semaphore_colors = getattr(prueba, 'semaphore_colors', None)
semaphore_values = getattr(prueba, 'semaphore_values', None)

# reconstruir climatologia (aplica filtros y suavizado N=30)
clim_uv = build_climatology_from_daily(df_daily_uv, vars_uv, N_smooth=30)

# -------------------------
# Interfaz de usuario (controles)
# -------------------------
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()
default_start = min_date
default_end = min(min_date + timedelta(days=30), max_date)

left, right = st.columns([3, 1])
with left:
    date_range = st.date_input("Rango de fechas", value=(default_start, default_end),
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple):
        date_start, date_end = date_range
    else:
        date_start = date_range
        date_end = date_range

    points_var = st.selectbox("Variable para puntos (observaciones)", vars_uv, index=0)
    clim_var = st.selectbox("Variable de la climatología (media suavizada, N=30)", vars_uv, index=0)

    # nuevo: agregado (cada valor / mean / max / min diario)
    agg_choice = st.selectbox("Mostrar puntos como:", ['cada valor', 'promedio diario', 'máximo diario', 'mínimo diario'])

    # nuevo: tipo de punto / line style
    point_style = st.selectbox("Tipo de marcador/linea:",
                               ['semaphore (coloreado por índice)', 'punto simple', 'línea', 'línea + punto'])

    # nuevo: estilo de figura
    style_choice = st.selectbox("Estilo de visualización:", ['Estilo actual', 'Estilo académico'])

    # mostrar elementos
    show_points = st.checkbox("Mostrar puntos/serie (según agregado)", value=True)
    show_clim_mean = st.checkbox("Mostrar media suavizada de la climatología", value=True)
    show_clim_band = st.checkbox("Mostrar banda ±2σ (std suavizada)", value=True)

with right:
    st.markdown("**Exportar / Descargas**")
    st.write(f"N_smooth (climatología) = **{N_smooth}**")
    st.write(f"Fechas overpass disponibles: {min_date} → {max_date}")

# -------------------------
# Preparar datos a plotear según agregado
# -------------------------
fecha_inicio = pd.to_datetime(date_start)
fecha_fin = pd.to_datetime(date_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
fechas = pd.date_range(fecha_inicio.date(), fecha_fin.date(), freq='D')

def doy_to_index(doy):
    return 365 if doy == 366 else int(doy)

# construir serie climatologia (mean_smooth) y sigma (std_smooth)
def build_clim_series_for_dates(clim_df, fechas, var):
    out = []
    for d in fechas.dayofyear:
        doy = doy_to_index(d)
        col = f"{var}_mean_smooth"
        if col in clim_df.columns and doy in clim_df.index:
            out.append(clim_df.at[doy, col])
        else:
            col2 = f"{var}_mean"
            out.append(clim_df.at[doy, col2] if col2 in clim_df.columns and doy in clim_df.index else np.nan)
    return np.array(out, dtype=float)

def build_clim_std_for_dates(clim_df, fechas, var):
    out = []
    for d in fechas.dayofyear:
        doy = doy_to_index(d)
        col = f"{var}_std_smooth"
        if col in clim_df.columns and doy in clim_df.index:
            out.append(clim_df.at[doy, col])
        else:
            col2 = f"{var}_std"
            out.append(clim_df.at[doy, col2] if col2 in clim_df.columns and doy in clim_df.index else np.nan)
    return np.array(out, dtype=float)

# -------------------------
# Helpers de color / mapeo
# -------------------------
def map_color_vals(vals, semaphore_colors=semaphore_colors, semaphore_values=semaphore_values):
    """
    Mapear una lista/array de valores numéricos a colores.
    - Si existe semaphore_colors y semaphore_values se usa esa mapping (espera índices enteros).
    - Sino, usa una paleta de fallback.
    Devuelve una lista de hex colors con la misma longitud que vals.
    """
    cols = []
    # fallback palette (long enough)
    fallback = ['#2D6A4F', '#52B788', '#A7C957', '#FFD166', '#EF476F', '#9D4EDD', '#1f77b4', '#ff7f0e']

    for v in vals:
        try:
            if pd.isna(v):
                cols.append('#999999')
                continue
        except Exception:
            cols.append('#999999')
            continue

        # intentar usar semaphore mapping si está definido correctamente
        try:
            if semaphore_colors is not None and semaphore_values is not None:
                # redondear el valor a entero y mapear respecto al rango de semaphore_values
                idx = int(np.rint(v))
                minv = min(semaphore_values)
                pos = idx - minv
                pos = int(np.clip(pos, 0, len(semaphore_colors) - 1))
                cols.append(semaphore_colors[pos])
                continue
        except Exception:
            pass

        # fallback simple: truncar a entero y elegir de fallback
        try:
            ii = int(np.clip(int(np.floor(v if not pd.isna(v) else 0)), 0, len(fallback) - 1))
        except Exception:
            ii = 0
        cols.append(fallback[ii])

    return cols

# preparar los puntos según agg_choice
mask = (df['Datetime'] >= fecha_inicio) & (df['Datetime'] <= fecha_fin)
df_range = df.loc[mask].copy()

if agg_choice == 'cada valor':
    # cada overpass se usa tal cual
    plot_df = df_range.copy()
    plot_df['PlotDate'] = plot_df['Datetime'].dt.floor('D')  # keep full datetime for x, but also store day
    plot_x = plot_df['Datetime']
    plot_y = plot_df[points_var]
    # customdata similar a antes
    customdata = np.stack([
        plot_df.get('SZA', pd.Series(np.nan, index=plot_df.index)).fillna(np.nan).values,
        plot_df.get('SZA_noon_local', pd.Series(np.nan, index=plot_df.index)).fillna(np.nan).values,
        # mu and mu+2sigma from selected climatology for the DOY of each overpass
        [ (clim_uv.at[doy_to_index(int(d)), f"{clim_var}_mean_smooth"] if (f"{clim_var}_mean_smooth" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index) else (clim_uv.at[doy_to_index(int(d)), f"{clim_var}_mean"] if f"{clim_var}_mean" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index else np.nan))
          if not pd.isna(d) else np.nan for d in plot_df['DOY']],
        [ ( (clim_uv.at[doy_to_index(int(d)), f"{clim_var}_mean_smooth"] if (f"{clim_var}_mean_smooth" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index) else (clim_uv.at[doy_to_index(int(d)), f"{clim_var}_mean"] if f"{clim_var}_mean" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index else np.nan)) + 
            2*(clim_uv.at[doy_to_index(int(d)), f"{clim_var}_std_smooth"] if (f"{clim_var}_std_smooth" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index) else (clim_uv.at[doy_to_index(int(d)), f"{clim_var}_std"] if f"{clim_var}_std" in clim_uv.columns and doy_to_index(int(d)) in clim_uv.index else np.nan))
          ) if not pd.isna(d) else np.nan for d in plot_df['DOY']]
    ], axis=1)

    # colors
    def map_color_vals(vals):
        cols = []
        for v in vals:
            try:
                if np.isnan(v):
                    cols.append('#999999'); continue
            except:
                cols.append('#999999'); continue
            if semaphore_colors is not None and semaphore_values is not None:
                try:
                    idx = int(np.rint(v))
                    minv = min(semaphore_values)
                    pos = idx - minv
                    pos = int(np.clip(pos, 0, len(semaphore_colors)-1))
                    cols.append(semaphore_colors[pos]); continue
                except:
                    pass
            # fallback palette
            palette = ['#2D6A4F','#52B788','#A7C957','#FFD166','#EF476F','#9D4EDD']
            try:
                ii = int(np.clip(int(np.floor(v if not pd.isna(v) else 0)), 0, len(palette)-1))
            except:
                ii = 0
            cols.append(palette[ii])
        return cols

    plot_colors = map_color_vals(plot_y.fillna(0).values)

else:
    # Resample diario y agregar (mean/max/min)
    if df_range.empty:
        plot_df = pd.DataFrame(columns=['Datetime', points_var])
        plot_x = pd.Series(pd.to_datetime([]))
        plot_y = pd.Series(dtype=float)
        plot_colors = []
        customdata = np.empty((0,4))
    else:
        tmp = df_range.set_index('Datetime')[points_var].resample('D')
        if agg_choice == 'promedio diario':
            s = tmp.mean()
        elif agg_choice == 'máximo diario':
            s = tmp.max()
        elif agg_choice == 'mínimo diario':
            s = tmp.min()
        else:
            s = tmp.mean()

        # s is a Series indexed by date
        plot_x = s.index
        plot_y = s.values

        # calculamos mu (climatología) y mu+2sigma para cada fecha en plot_x (serie diaria agregada)
        mu_list = []
        mu2_list = []

        for d in plot_x:
            # asegurar que trabajamos con un Timestamp
            ts = pd.Timestamp(d)
            doy = doy_to_index(ts.dayofyear)

            # default
            mu = np.nan
            sigma = np.nan

            # sólo intentar leer si el DOY existe en clim_uv
            if doy in clim_uv.index:
                mean_s_col = f"{clim_var}_mean_smooth"
                mean_col = f"{clim_var}_mean"
                std_s_col = f"{clim_var}_std_smooth"
                std_col = f"{clim_var}_std"

                if mean_s_col in clim_uv.columns:
                    mu = clim_uv.at[doy, mean_s_col]
                elif mean_col in clim_uv.columns:
                    mu = clim_uv.at[doy, mean_col]

                if std_s_col in clim_uv.columns:
                    sigma = clim_uv.at[doy, std_s_col]
                elif std_col in clim_uv.columns:
                    sigma = clim_uv.at[doy, std_col]

            mu_list.append(mu)
            mu2_list.append(mu + 2*sigma if (not pd.isna(mu) and not pd.isna(sigma)) else np.nan)

        # customdata: SZA, SZA_noon_local (NaN para agregados diarios), mu, mu+2sigma
        customdata = np.stack([
            np.full(len(plot_x), np.nan),                      # SZA (no disponible al agregar diario)
            np.full(len(plot_x), np.nan),                      # SZA_noon_local (no disponible)
            np.array(mu_list, dtype=float),
            np.array(mu2_list, dtype=float)
        ], axis=1)

        # map colors sobre plot_y (serie agregada)
        plot_colors = map_color_vals(plot_y)


# -------------------------
# Construir figura Plotly con estilo elegido
# -------------------------
fig = go.Figure()

# styles
if style_choice == 'Estilo académico':
    template = 'plotly_white'
    font_family = "Times New Roman, Times, serif"
    title_font_size = 20
    axis_title_font_size = 14
    line_width_clim = 2.5
    marker_size = 8
else:
    template = None
    font_family = "Arial, sans-serif"
    title_font_size = 16
    axis_title_font_size = 12
    line_width_clim = 2
    marker_size = 9

# climatologia trace
if show_clim_mean:
    y_clim = build_clim_series_for_dates(clim_uv, fechas, clim_var)
    fig.add_trace(go.Scatter(x=fechas, y=y_clim, mode='lines',
                             name=f'Climatología (mean_smooth) — {clim_var}',
                             line=dict(color='gray', width=line_width_clim),
                             hovertemplate="Fecha: %{x}<br>Climatología: %{y:.3f}<extra></extra>"))

# banda ±2σ
if show_clim_band:
    mu = build_clim_series_for_dates(clim_uv, fechas, clim_var)
    sigma = build_clim_std_for_dates(clim_uv, fechas, clim_var)
    upper = mu + 2 * sigma
    lower = mu - 2 * sigma
    fig.add_trace(go.Scatter(x=fechas, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fechas, y=lower, mode='lines', fill='tonexty',
                             fillcolor='rgba(200,200,200,0.25)', line=dict(width=0),
                             name='Banda ±2σ', hoverinfo='skip'))

# puntos/linea según elección
if show_points:
    # determine mode
    if point_style == 'semaphore (coloreado por índice)':
        mode = 'markers'
        marker_colors = plot_colors
        line_mode = None
        if agg_choice in ['cada valor'] and 'línea' in point_style:
            pass
    elif point_style == 'punto simple':
        mode = 'markers'
        marker_colors = ['#1f77b4'] * len(plot_colors) if len(plot_colors)>0 else None
    elif point_style == 'línea':
        mode = 'lines'
        marker_colors = None
    else:  # 'línea + punto'
        mode = 'lines+markers'
        marker_colors = plot_colors if point_style.startswith('semaphore') else (['#1f77b4'] * len(plot_colors) if len(plot_colors)>0 else None)

    # add trace
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode=mode,
        name=f'{points_var} ({agg_choice})',
        marker=dict(size=marker_size, color=marker_colors, line=dict(width=0.5, color='black')),
        line=dict(width=2),
        hovertemplate=(
            "Fecha: %{x}<br>"
            f"{points_var}: "+"%{y:.3f}<br>"
            "SZA_overpass: %{customdata[0]:.2f}<br>"
            "SZA_noon_local: %{customdata[1]:.2f}<br>"
            f"mu({clim_var}): "+"%{customdata[2]:.3f}<br>"
            f"mu+2σ({clim_var}): "+"%{customdata[3]:.3f}<extra></extra>"
        ),
        customdata=customdata if 'customdata' in locals() else None
    ))

# layout adjustments
fig.update_layout(
    template=template,
    font=dict(family=font_family, size=12),
    title=dict(text=f"Climatología: {clim_var} (N={N_smooth}) — Observaciones: {points_var} ({agg_choice})", x=0.01, xanchor='left', font=dict(size=title_font_size)),
    xaxis_title="Fecha",
    yaxis_title="Índice UV",
    hovermode="closest",
    height=650
)

# eje Y estilo académico
if style_choice == 'Estilo académico':
    fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=axis_title_font_size), gridcolor='lightgray', zeroline=True)
    fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=axis_title_font_size), gridcolor='lightgray')

# mostrar figura
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Export / Download buttons
# -------------------------
col_png, col_html, col_csv = st.columns([1,1,1])
with col_png:
    st.write("Guardar imagen (PNG):")
    try:
        # usar kaleido para render PNG
        img_bytes = pio.to_image(fig, format='png', engine='kaleido')
        st.download_button("Descargar PNG", data=img_bytes, file_name="uvi_plot.png", mime="image/png")
    except Exception as e:
        st.warning("Exportar PNG requiere 'kaleido' en requirements. Si está instalado y falla, revisá: " + str(e))

with col_html:
    html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    st.download_button("Descargar HTML interactivo", data=html, file_name="uvi_plot.html", mime="text/html")

with col_csv:
    # generar CSV de los datos mostrados (plot)
    if agg_choice == 'cada valor':
        csv_df = plot_df[['Datetime', points_var]].copy()
        csv_df = csv_df.rename(columns={points_var: 'value'}).sort_values('Datetime')
    else:
        csv_df = pd.DataFrame({'Datetime': plot_x, 'value': plot_y})
    st.download_button("Descargar datos (CSV)", data=csv_df.to_csv(index=False), file_name="uvi_plot_data.csv", mime="text/csv")

# tabla de observaciones (rango)
st.markdown("### Tabla: observaciones (rango seleccionado)")
display_cols = ['Datetime', 'DOY', 'SZA']
if 'SZA_noon_local' in df.columns:
    display_cols.append('SZA_noon_local')
if points_var in df.columns:
    display_cols.append(points_var)
st.dataframe(df.loc[(df['Datetime'] >= fecha_inicio) & (df['Datetime'] <= fecha_fin)][display_cols].sort_values('Datetime').reset_index(drop=True))