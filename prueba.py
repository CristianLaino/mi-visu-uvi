#%%
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# -------------------------
# 0. Parámetros
# -------------------------
url = (
    "https://avdc.gsfc.nasa.gov/pub/data/satellite/"
    "Aura/OMI/V03/L2OVP/OMUVB/"
    "aura_omi_l2ovp_omuvb_v03_rio.gallegos.txt"
)
cols = [
    "Datetime","MJD2000","Year","DOY","sec.(UT)","Orbit","CTP",
    "Lat.","Lon.","Dist","SZA","VZA","GPQF","OMAF","OMQF","UVBQF",
    "OMTO3_O3","CSEDDose","CSEDRate","CSIrd305","CSIrd310","CSIrd324",
    "CSIrd380","CldOpt","EDDose","EDRate","Ird305","Ird310","Ird324",
    "Ird380","OPEDRate","OPIrd305","OPIrd310","OPIrd324","OPIrd380",
    "CSUVindex","OPUVindex","UVindex","LambEquRef","SufAlbedo","TerrHgt"
]
vars_uv = ['CSUVindex', 'OPUVindex', 'UVindex', 'UVI_Madronich']
min_years = 0   # poner >0 si querés filtrar DOY con pocos años
N_smooth = 30   # ventana de suavizado para climatología
window = 15     # ventana ± días para percentil empírico (total 2*window)
date_start = '2025-03-23'
date_end   = '2025-03-25'

# -------------------------
# 1. CARGA DE DATOS (OMI)
# -------------------------
with urllib.request.urlopen(url) as response:
    lines = response.read().decode('utf-8').split('\n')[51:]
data = [ln.split() for ln in lines if ln]
df = pd.DataFrame(data, columns=cols)

# Conversión de tipos
df['Datetime'] = pd.to_datetime(df['Datetime'].str[:8], format='%Y%m%d')
df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric, errors='coerce')
df['DOY'] = df['Datetime'].dt.dayofyear
df.loc[df['DOY'] == 366, 'DOY'] = 365

# -------------------------
# 2. SZA mediodía local
# -------------------------
lat = -51.60  # Río Gallegos

def declinacion_solar(doy):
    return np.radians(23.45) * np.sin(np.radians((360/365)*(284 + doy)))

def sza_mediodia_local(lat, doy):
    phi = np.radians(lat)
    delta = declinacion_solar(doy)
    H = 0
    cos_theta_z = np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H)
    cos_theta_z = np.clip(cos_theta_z, -1, 1)
    return np.degrees(np.arccos(cos_theta_z))

df['SZA_noon_local'] = df['DOY'].apply(lambda doy: sza_mediodia_local(lat, doy))

# -------------------------
# 3. UVI_Madronich (fórmula analítica) usando SZA overpass
# -------------------------
coef = [13.1246249017015, 2.68422133639481, -1.23515289658719]
mu_obs = np.cos(np.radians(df['SZA'])).clip(lower=0)
toc_obs = df['OMTO3_O3'].where(df['OMTO3_O3'] > 0, np.nan)
df['UVI_Madronich'] = coef[0] * mu_obs**coef[1] * (toc_obs/300)**coef[2]

# -------------------------
# 4. Serie diaria y climatología por DOY (suavizada)
# -------------------------
df_daily_uv = df.set_index('Datetime')[vars_uv].resample('D').mean()
df_daily_uv = df_daily_uv.dropna(how='all')
df_daily_uv['DOY'] = df_daily_uv.index.dayofyear

# climatología: mean, std, count por DOY
clim_uv = df_daily_uv.groupby('DOY')[vars_uv].agg(['mean','std','count'])
clim_uv.columns = [f"{var}_{stat}" for var, stat in clim_uv.columns]

# filtrar por min_years (opcional)
if min_years > 0:
    for var in vars_uv:
        mask = clim_uv[f"{var}_count"] < min_years
        clim_uv.loc[mask, f"{var}_mean"] = np.nan
        clim_uv.loc[mask, f"{var}_std"] = np.nan

# suavizado rolling
for var in vars_uv:
    clim_uv[f"{var}_mean_smooth"] = (
        clim_uv[f"{var}_mean"].rolling(window=N_smooth, center=True, min_periods=1).mean()
    )
    clim_uv[f"{var}_std_smooth"] = (
        clim_uv[f"{var}_std"].rolling(window=N_smooth, center=True, min_periods=1).mean()
    )

clim_uv = clim_uv.sort_index()  # índice DOY
#%%
# -------------------------
# 5. Recalcular tabla de estadísticas usando la climatología suavizada
# -------------------------
def calcular_estadisticas_con_climatologia_suavizada(df_overpass, df_daily_uv, clim_uv, vars_uv, ventana=15):
    registros = []
    df_daily = df_daily_uv.copy()
    df_daily['DOY'] = df_daily.index.dayofyear

    for _, row in df_overpass.iterrows():
        fecha = row['Datetime']
        doy = int(row['DOY'])

        for var in vars_uv:
            obs_val = row.get(var, np.nan)
            if pd.isna(obs_val):
                continue

            mean_col = f"{var}_mean_smooth"
            std_col = f"{var}_std_smooth"
            if mean_col not in clim_uv.columns or std_col not in clim_uv.columns:
                continue

            # obtener mu y sigma suavizados para ese DOY (si no existe dará KeyError -> protegemos)
            try:
                mu = clim_uv.at[doy, mean_col]
                sigma = clim_uv.at[doy, std_col]
            except Exception:
                mu = np.nan
                sigma = np.nan

            if pd.isna(mu) or pd.isna(sigma):
                continue

            mu_2sigma = mu + 2*sigma

            # ventana circular sobre serie DIARIA para percentil empírico
            half = ventana
            diff = (df_daily['DOY'] - doy + 365) % 365
            mask = (diff <= half) | (diff >= 365 - half)
            window_data = df_daily.loc[mask, var].dropna()

            if window_data.empty:
                continue

            percentil_emp = 100.0 * (window_data <= obs_val).mean()
            delta_mu_pct = 100.0 * (obs_val - mu) / mu if (mu != 0 and not pd.isna(mu)) else np.nan
            delta_mu_2sigma_pct = 100.0 * (obs_val - mu_2sigma) / mu_2sigma if (mu_2sigma != 0 and not pd.isna(mu_2sigma)) else np.nan
            z = (obs_val - mu) / sigma if (sigma > 0 and not pd.isna(sigma)) else np.nan

            registros.append({
                "Fecha": fecha,
                "Variable": var,
                "UVI_obs": obs_val,
                "SZA_overpass": row.get('SZA', np.nan),
                "SZA_noon_local": row.get('SZA_noon_local', np.nan),
                "mu": mu,
                "mu+2sigma": mu_2sigma,
                "delta_mu_%": delta_mu_pct,
                "delta_mu+2sigma_%": delta_mu_2sigma_pct,
                "percentil_emp_window": percentil_emp,
                "z": z
            })
    return pd.DataFrame(registros)

resultados_clim = calcular_estadisticas_con_climatologia_suavizada(df, df_daily_uv, clim_uv, vars_uv, ventana=window)

# Filtrar período de interés
mask_period = (resultados_clim['Fecha'] >= pd.to_datetime(date_start)) & (resultados_clim['Fecha'] <= pd.to_datetime(date_end))
resultados_mar2025 = resultados_clim.loc[mask_period].sort_values(['Variable','Fecha']).reset_index(drop=True)

# Mostrar y guardar
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)
print(resultados_mar2025)
resultados_mar2025.to_csv('resultados_mar2025_con_climatologia_diaria_suavizada.csv', index=False)
print("Tabla guardada en 'resultados_mar2025_con_climatologia_diaria_suavizada.csv'")
#%%
import matplotlib.patches as mpatches

# Defino semáforo
semaphore_colors = [
    '#658D1B', '#84BD00', '#97D700', '#F7EA48', '#FCE300', '#FFCD00', '#ECA154',
    '#FF8200', '#EF3340', '#DA291C', '#BF0D3E', '#4B1E88', '#62359F', '#794CB6',
    '#9063CD', '#A77AE4', '#BE91FB', '#D5A8FF', '#ECBFFF', '#FFD6FF', '#FFEDFF', '#FFFFFF'
]
semaphore_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# función auxiliar para obtener color según valor UVI
def color_from_uvi(value, semaphore_values, semaphore_colors):
    if np.isnan(value):
        return '#999999'  # gris para NaN
    # redondear al entero más cercano y clippear al rango disponible
    idx = int(np.rint(value))
    idx = int(np.clip(idx, min(semaphore_values), max(semaphore_values)))
    # idx corresponde al valor; obtener posición en la lista (siempre coinciden índices aquí)
    # asumimos semaphore_values = [0,1,2,...] en orden; si no, habría que buscar index()
    pos = idx - semaphore_values[0]
    pos = int(np.clip(pos, 0, len(semaphore_colors) - 1))
    return semaphore_colors[pos]
#%%
# versión actualizada de la función de graficado
def graficar_clim_con_obs_colored(clim_uv, resultados_df, df_daily_uv, variables, fecha_inicio, fecha_fin,
                                 semaphore_values, semaphore_colors):
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq='D')
    doys = fechas.dayofyear

    for var in variables:
        # construir series de climatologia para las fechas (puede haber NaN)
        mu_series = [clim_uv.at[doy, f"{var}_mean_smooth"] if f"{var}_mean_smooth" in clim_uv.columns and doy in clim_uv.index else np.nan for doy in doys]
        std_series = [clim_uv.at[doy, f"{var}_std_smooth"] if f"{var}_std_smooth" in clim_uv.columns and doy in clim_uv.index else np.nan for doy in doys]
        mu_series = np.array(mu_series, dtype=float)
        std_series = np.array(std_series, dtype=float)
        upper = mu_series + 2*std_series
        lower = mu_series - 2*std_series

        # observaciones overpass en el periodo
        obs = resultados_df[resultados_df['Variable'] == var].copy()
        obs = obs[(obs['Fecha'] >= pd.to_datetime(fecha_inicio)) & (obs['Fecha'] <= pd.to_datetime(fecha_fin))]

        plt.figure(figsize=(9,5))
        # climatología suavizada y bandas (estilo tuyos)
        plt.plot(fechas, mu_series, linestyle='-', color='gray', label='Media móvil')
        plt.fill_between(fechas, lower, upper, color ='lightgray', alpha=0.25, label='±2σ')

        # graficar observaciones overpass con colores del semaforo
        if not obs.empty:
            # calcular color para cada punto
            colors = [color_from_uvi(v, semaphore_values, semaphore_colors) for v in obs['UVI_obs'].values]

            # separar extremos (por encima de mu+2sigma) para marcarlos con X más grande
            if 'mu' in obs.columns and 'mu+2sigma' in obs.columns:
                ext_mask = obs['UVI_obs'] > obs['mu+2sigma']
                nonext = obs.loc[~ext_mask]
                ext = obs.loc[ext_mask]
                if not nonext.empty:
                    colors_nonext = [color_from_uvi(v, semaphore_values, semaphore_colors) for v in nonext['UVI_obs'].values]
                    plt.scatter(nonext['Fecha'], nonext['UVI_obs'], s=80, marker='o', edgecolor='k', linewidth=0.5,
                                c=colors_nonext, label='UVI overpass')
                if not ext.empty:
                    colors_ext = [color_from_uvi(v, semaphore_values, semaphore_colors) for v in ext['UVI_obs'].values]
                    plt.scatter(ext['Fecha'], ext['UVI_obs'], s=140, marker='X', edgecolor='k', linewidth=0.8,
                                c=colors_ext, label='UVI > mu+2σ')
            else:
                plt.scatter(obs['Fecha'], obs['UVI_obs'], s=80, marker='o', edgecolor='k', linewidth=0.5,
                            c=colors, label='UVI overpass')

        plt.title(f'{var} — {fecha_inicio} a {fecha_fin}')
        plt.xlabel('Fecha')
        plt.ylabel('Índice UV')
        plt.ylim(0,10)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # construir una leyenda compacta del semáforo: tomar un subconjunto cada 2 valores
        legend_indices = list(range(0, len(semaphore_values), 2))
        legend_patches = []
        for ii in legend_indices:
            val = semaphore_values[ii]
            col = semaphore_colors[ii]
            patch = mpatches.Patch(color=col, label=str(val))
            legend_patches.append(patch)
        # añadir la leyenda del semáforo a la derecha
        plt.legend(handles=legend_patches + plt.gca().get_legend_handles_labels()[0], loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.show()

# Llamada a la función (usa tus variables definidas)
graficar_clim_con_obs_colored(clim_uv, resultados_mar2025, df_daily_uv, vars_uv, date_start, date_end,
                             semaphore_values, semaphore_colors)



# %%
