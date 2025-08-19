# app_streamlit.py
"""
Visor UVI — múltiples variables para puntos y climatologías
- Selección múltiple de variables para observaciones (points_vars)
- Selección múltiple de variables para climatologías (clim_vars)
- Mantiene agregado (cada valor / mean / max / min diario), estilos y exportaciones
- Depende de prueba.py (df, df_daily_uv, vars_uv, opcionales: N_smooth, semaphore_colors, semaphore_values)
"""

import importlib
from datetime import timedelta
import io

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as plc

MODULE_NAME = "prueba"

# -------------------------
# Load / reload prueba.py safely (module object -> use cache_resource)
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
# Build climatology (dropna, percentile filter, group by DOY, smooth)
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
# Helpers
# -------------------------
def doy_to_index(doy):
    return 365 if doy == 366 else int(doy)

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

def map_color_vals(vals, semaphore_colors=None, semaphore_values=None):
    """
    Map numeric values to hex colors. If semaphore mapping is provided, use it.
    Otherwise use a fallback qualitative palette repeated/clipped.
    """
    cols = []
    fallback = plc.qualitative.Plotly
    for v in vals:
        try:
            if pd.isna(v):
                cols.append('#999999'); continue
        except Exception:
            cols.append('#999999'); continue
        if semaphore_colors is not None and semaphore_values is not None:
            try:
                idx = int(np.rint(v))
                minv = min(semaphore_values)
                pos = idx - minv
                pos = int(np.clip(pos, 0, len(semaphore_colors)-1))
                cols.append(semaphore_colors[pos])
                continue
            except Exception:
                pass
        # fallback: map integer part to fallback palette
        try:
            ii = int(np.clip(int(np.floor(v if not pd.isna(v) else 0)), 0, len(fallback)-1))
        except Exception:
            ii = 0
        cols.append(fallback[ii])
    return cols

# -------------------------
# Start app
# -------------------------
st.set_page_config(page_title="Visor UVI — Multi-variable", layout="wide")
st.title("Visor UVI — múltiples variables para puntos y climatologías")

# reload button
reload_col, info_col = st.columns([1,4])
with reload_col:
    if st.button("🔄 Recargar prueba.py (force)"):
        try:
            prueba = force_reload_prueba()
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error recargando: {e}")

with info_col:
    st.markdown("Seleccioná una o más variables para las observaciones y/o para la climatología. Cada variable se graficará con su propia traza.")

# load module
try:
    prueba = load_prueba_module()
except Exception as e:
    st.error(f"No se pudo importar '{MODULE_NAME}': {e}")
    st.stop()

# required objects
required = ['df', 'df_daily_uv', 'vars_uv']
missing = [r for r in required if not hasattr(prueba, r)]
if missing:
    st.error(f"prueba.py debe contener las variables: {missing}")
    st.stop()

# get objects
df = prueba.df.copy()
df_daily_uv = prueba.df_daily_uv.copy()
vars_uv = list(prueba.vars_uv)
N_smooth = getattr(prueba, 'N_smooth', 30)
semaphore_colors = getattr(prueba, 'semaphore_colors', None)
semaphore_values = getattr(prueba, 'semaphore_values', None)

# build climatology
clim_uv = build_climatology_from_daily(df_daily_uv, vars_uv, N_smooth=N_smooth if N_smooth is not None else 30)

# UI controls
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()
default_start = min_date
default_end = min(min_date + timedelta(days=30), max_date)

left, right = st.columns([3,1])
with left:
    date_range = st.date_input("Rango de fechas", value=(default_start, default_end),
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple):
        date_start, date_end = date_range
    else:
        date_start = date_range
        date_end = date_range

    # MULTISELECTS
    points_vars = st.multiselect("Variables para puntos (observaciones) — puede elegir varias", vars_uv, default=[vars_uv[0]])
    clim_vars = st.multiselect("Variables para climatología (media suavizada) — puede elegir varias", vars_uv, default=[vars_uv[0]])

    agg_choice = st.selectbox("Mostrar puntos como:", ['cada valor', 'promedio diario', 'máximo diario', 'mínimo diario'])
    point_style = st.selectbox("Tipo de marcador/linea:", ['semaphore (coloreado por índice)', 'punto simple', 'línea', 'línea + punto'])
    style_choice = st.selectbox("Estilo de visualización:", ['Estilo actual', 'Estilo académico'])

    show_points = st.checkbox("Mostrar observaciones/series", value=True)
    show_clim_mean = st.checkbox("Mostrar media suavizada de la(s) climatología(es)", value=True)
    show_clim_band = st.checkbox("Mostrar banda ±2σ para las climatologías", value=True)

with right:
    st.markdown("**Export / Info**")
    st.write(f"N_smooth (climatología) = **{N_smooth}**")
    st.write(f"Fechas overpass: {min_date} → {max_date}")
    st.download_button("Descargar overpass procesado (CSV)", df.to_csv(index=False), file_name="overpass_processed.csv")

# prepare date range
fecha_inicio = pd.to_datetime(date_start)
fecha_fin = pd.to_datetime(date_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
fechas = pd.date_range(fecha_inicio.date(), fecha_fin.date(), freq='D')

# prepare figure
fig = go.Figure()

# styling params
if style_choice == 'Estilo académico':
    template = 'plotly_white'
    font_family = "Times New Roman, Times, serif"
    title_font_size = 18
    axis_title_font_size = 14
    line_width_clim = 2.5
    marker_size = 8
else:
    template = None
    font_family = "Arial, sans-serif"
    title_font_size = 14
    axis_title_font_size = 12
    line_width_clim = 2
    marker_size = 9

# plot climatologies (multiple)
color_cycle = plc.qualitative.Plotly
if show_clim_mean and clim_vars:
    for i, cv in enumerate(clim_vars):
        color = color_cycle[i % len(color_cycle)]
        y_clim = build_clim_series_for_dates(clim_uv, fechas, cv)
        fig.add_trace(go.Scatter(x=fechas, y=y_clim, mode='lines',
                                 name=f'Clim: {cv}', line=dict(color=color, width=line_width_clim),
                                 hovertemplate="Fecha: %{x}<br>Climatología: %{y:.3f}<extra></extra>"))
        # band
        if show_clim_band:
            sigma = build_clim_std_for_dates(clim_uv, fechas, cv)
            upper = y_clim + 2*sigma
            lower = y_clim - 2*sigma
            fig.add_trace(go.Scatter(x=fechas, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=fechas, y=lower, mode='lines', fill='tonexty',
                                     fillcolor='rgba(200,200,200,0.15)', line=dict(width=0),
                                     name=f'Banda ±2σ {cv}', hoverinfo='skip'))

# Helper to get per-variable series for plotting (handles agg_choice)
def get_series_for_var(var):
    mask = (df['Datetime'] >= fecha_inicio) & (df['Datetime'] <= fecha_fin)
    df_range = df.loc[mask].copy()
    if agg_choice == 'cada valor':
        x = df_range['Datetime']
        y = df_range[var]
        # customdata: SZA, SZA_noon_local, mu, mu+2sigma (mu from climatology for DOY of each observation)
        mu_vals = []
        mu2_vals = []
        for doy in df_range['DOY']:
            if pd.isna(doy):
                mu_vals.append(np.nan); mu2_vals.append(np.nan); continue
            d = doy_to_index(int(doy))
            mean_col = f"{var}_mean_smooth" if (f"{var}_mean_smooth" in clim_uv.columns) else f"{var}_mean"
            std_col = f"{var}_std_smooth" if (f"{var}_std_smooth" in clim_uv.columns) else f"{var}_std"
            mu = clim_uv.at[d, mean_col] if (mean_col in clim_uv.columns and d in clim_uv.index) else np.nan
            sigma = clim_uv.at[d, std_col] if (std_col in clim_uv.columns and d in clim_uv.index) else np.nan
            mu_vals.append(mu)
            mu2_vals.append(mu + 2*sigma if (not pd.isna(mu) and not pd.isna(sigma)) else np.nan)
        custom = np.stack([
            df_range.get('SZA', pd.Series(np.nan, index=df_range.index)).fillna(np.nan).values,
            df_range.get('SZA_noon_local', pd.Series(np.nan, index=df_range.index)).fillna(np.nan).values,
            np.array(mu_vals, dtype=float),
            np.array(mu2_vals, dtype=float)
        ], axis=1)
        return x, y, custom
    else:
        # daily aggregate
        tmp = df_range.set_index('Datetime')[var].resample('D')
        if agg_choice == 'promedio diario':
            s = tmp.mean()
        elif agg_choice == 'máximo diario':
            s = tmp.max()
        elif agg_choice == 'mínimo diario':
            s = tmp.min()
        else:
            s = tmp.mean()
        x = s.index
        y = s.values
        # compute mu/mu2 per day
        mu_list = []
        mu2_list = []
        for d in x:
            ts = pd.Timestamp(d)
            doy = doy_to_index(ts.dayofyear)
            mu = np.nan; sigma = np.nan
            if doy in clim_uv.index:
                mean_s_col = f"{var}_mean_smooth"
                mean_col = f"{var}_mean"
                std_s_col = f"{var}_std_smooth"
                std_col = f"{var}_std"
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
        custom = np.stack([
            np.full(len(x), np.nan),
            np.full(len(x), np.nan),
            np.array(mu_list, dtype=float),
            np.array(mu2_list, dtype=float)
        ], axis=1)
        return x, y, custom

# Plot points / series for each selected variable
if show_points and points_vars:
    for j, pv in enumerate(points_vars):
        x, y, custom = get_series_for_var(pv)
        # determine mode and colors
        if point_style == 'semaphore (coloreado por índice)':
            mode = 'markers' if agg_choice == 'cada valor' else 'lines+markers'
            marker_colors = map_color_vals(y, semaphore_colors=semaphore_colors, semaphore_values=semaphore_values)
            line_color = None
        elif point_style == 'punto simple':
            mode = 'markers'
            marker_colors = None
            line_color = '#1f77b4'
        elif point_style == 'línea':
            mode = 'lines'
            marker_colors = None
            line_color = plc.qualitative.Plotly[j % len(plc.qualitative.Plotly)]
        else:  # 'línea + punto'
            mode = 'lines+markers'
            marker_colors = None
            line_color = plc.qualitative.Plotly[j % len(plc.qualitative.Plotly)]

        # Add trace (per variable)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode=mode,
            name=f'{pv} ({agg_choice})',
            marker=dict(size=marker_size, color=marker_colors, line=dict(width=0.4, color='black') if marker_colors is not None else dict(width=0)),
            line=dict(width=2, color=line_color),
            customdata=custom,
            hovertemplate=(
                "Fecha: %{x}<br>"
                f"{pv}: "+"%{y:.3f}<br>"
                "SZA_overpass: %{customdata[0]:.2f}<br>"
                "SZA_noon_local: %{customdata[1]:.2f}<br>"
                "mu: %{customdata[2]:.3f}<br>"
                "mu+2σ: %{customdata[3]:.3f}<extra></extra>"
            )
        ))

# Layout
fig.update_layout(
    template=template,
    font=dict(family=font_family, size=12),
    title=dict(text=f"Climatologías: {', '.join(clim_vars)} — Observaciones: {', '.join(points_vars)}", x=0.01, xanchor='left', font=dict(size=title_font_size)),
    xaxis_title="Fecha",
    yaxis_title="Índice UV",
    hovermode="closest",
    height=700
)

if style_choice == 'Estilo académico':
    fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=axis_title_font_size), gridcolor='lightgray', zeroline=True)
    fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=axis_title_font_size), gridcolor='lightgray')

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Export / download: combined CSV, HTML, PNG
# -------------------------
# Build CSV with combined data for selected variables
def build_combined_csv(points_vars, agg_choice, fecha_inicio, fecha_fin):
    mask = (df['Datetime'] >= fecha_inicio) & (df['Datetime'] <= fecha_fin)
    df_range = df.loc[mask].copy()
    if not points_vars:
        return pd.DataFrame()
    if agg_choice == 'cada valor':
        out = df_range[['Datetime','DOY'] + points_vars].copy()
        return out.sort_values('Datetime').reset_index(drop=True)
    else:
        resampled = []
        for pv in points_vars:
            tmp = df_range.set_index('Datetime')[pv].resample('D')
            if agg_choice == 'promedio diario':
                s = tmp.mean()
            elif agg_choice == 'máximo diario':
                s = tmp.max()
            elif agg_choice == 'mínimo diario':
                s = tmp.min()
            else:
                s = tmp.mean()
            s = s.rename(pv).reset_index()
            resampled.append(s)
        # merge on Datetime
        dfm = resampled[0]
        for dfp in resampled[1:]:
            dfm = dfm.merge(dfp, on='Datetime', how='outer')
        return dfm.sort_values('Datetime').reset_index(drop=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    csv_df = build_combined_csv(points_vars, agg_choice, fecha_inicio, fecha_fin)
    st.download_button("Descargar datos (CSV)", data=csv_df.to_csv(index=False), file_name="uvi_plot_data.csv", mime="text/csv")
with col2:
    html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    st.download_button("Descargar HTML interactivo", data=html, file_name="uvi_plot.html", mime="text/html")
with col3:
    try:
        img_bytes = pio.to_image(fig, format='png', engine='kaleido')
        st.download_button("Descargar PNG", data=img_bytes, file_name="uvi_plot.png", mime="image/png")
    except Exception as e:
        st.warning("Exportar PNG requiere 'kaleido' y un Chrome/Chromium disponible. Si falla, usá la descarga HTML. Detalle: " + str(e))

# Table of observations (range)
st.markdown("### Tabla: observaciones (rango seleccionado)")
display_cols = ['Datetime', 'DOY', 'SZA'] + [v for v in points_vars if v in df.columns]
if 'SZA_noon_local' in df.columns:
    display_cols.insert(3, 'SZA_noon_local')
if not display_cols:
    st.write("No hay columnas seleccionadas para mostrar.")
else:
    st.dataframe(df.loc[(df['Datetime'] >= fecha_inicio) & (df['Datetime'] <= fecha_fin)][display_cols].sort_values('Datetime').reset_index(drop=True))
