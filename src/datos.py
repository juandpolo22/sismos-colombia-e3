"""Carga, ETL y construcción del panel celda × mes.

Flujo de datos (ETL)
--------------------
1. **Extracción**: diez archivos descargados de los dos portales del SGC:
   ``reporte*.xlsx`` (portal histórico, 2004-01 a 2018-02, estado «Revisado») y
   ``Catalogo_Sismico*.csv`` (portal actual, 2018-03 a 2026-10, estados «manual» y «automatic»).
2. **Transformación**: se armonizan nombres, unidades y tipos; se conserva solo la solución revisada
   por un analista (``Revisado``/``manual``); se eliminan registros sin fecha, coordenadas o magnitud,
   duplicados exactos y sismos sin departamento colombiano (fronterizos o en el mar).
3. **Carga**: ``data/sismos_sgc_2004_2026.csv`` (una fila por sismo) y el panel celda × mes que usan
   los modelos (una fila por celda activa y mes; el objetivo es el conteo del mes siguiente).
"""
from __future__ import annotations

import glob
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as cfg


# ---------------------------------------------------------------------------
# 1. Extracción y armonización de los archivos crudos
# ---------------------------------------------------------------------------
def normalizar_texto(txt) -> str:
    """Mayúsculas sin tildes ni espacios repetidos (para nombres de departamento)."""
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return np.nan
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return " ".join(txt.upper().replace("_", " ").split())


def _leer_xlsx(ruta: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(ruta, engine="calamine")
    except Exception:  # noqa: BLE001  (algunos .xlsx del SGC traen estilos que openpyxl no lee)
        df = pd.read_excel(ruta)
    df.columns = [str(c).strip() for c in df.columns]
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    ml, mw = num("MAGNITUD Ml"), num("MAGNITUD Mw")
    return pd.DataFrame({
        "time": pd.to_datetime(df["FECHA"].astype(str).str[:10] + " " + df["HORA_UTC"].astype(str), errors="coerce"),
        "latitude": num("LATITUD (grados)"), "longitude": num("LONGITUD (grados)"),
        "depth": num("PROFUNDIDAD (Km)"), "mag": ml.fillna(mw),
        "tipo_mag": np.where(ml.notna(), "Ml", np.where(mw.notna(), "Mw", None)),
        "fases": num("# FASES"), "rms": num("RMS (Seg)"), "gap_az": num("GAP (grados)"),
        "err_lat": num("ERROR LATITUD (Km)"), "err_lon": num("ERROR LONGITUD (Km)"), "err_prof": num("ERROR PROFUNDIDAD (Km)"),
        "depto": df["DEPARTAMENTO"].map(normalizar_texto), "region": df["MUNICIPIO"].astype(str),
        "estado": df["ESTADO"].astype(str).str.strip().str.lower(), "portal": "histórico (xlsx)", "archivo": ruta.name,
    })


def _leer_csv(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    col_t = [c for c in df.columns if c.startswith("Fecha-Hora")][0]
    reg = df["Region"].astype(str)
    # «Municipio - Departamento, Colombia»; las regiones fronterizas o marinas no tienen departamento
    es_depto = reg.str.contains(" - ") & reg.str.contains("Colombia")
    depto = reg.str.replace(", Colombia", "", regex=False).str.split(" - ").str[-1].map(normalizar_texto)
    return pd.DataFrame({
        "time": pd.to_datetime(df[col_t], errors="coerce"),
        "latitude": num("Lat(°)"), "longitude": num("Long(°)"), "depth": num("Prof(Km)"), "mag": num("Mag."),
        "tipo_mag": df["Tipo Mag."].astype(str), "fases": num("Fases"), "rms": num("Rms(Seg)"), "gap_az": num("Gap(°)"),
        "err_lat": num("Error Lat(Km)"), "err_lon": num("Error Long(Km)"), "err_prof": num("Error Prof(Km)"),
        "depto": depto.where(es_depto), "region": reg,
        "estado": df["Estado"].astype(str).str.strip().str.lower(), "portal": "actual (csv)", "archivo": ruta.name,
    })


def leer_crudos(carpeta: Path | str = cfg.CRUDOS) -> pd.DataFrame:
    """Lee y armoniza los diez archivos crudos del SGC (sin filtrar nada).

    Returns
    -------
    pandas.DataFrame
        Una fila por registro del portal, con columnas homogéneas (``time``, ``latitude``, ``longitude``,
        ``depth``, ``mag``, ``tipo_mag``, ``fases``, ``rms``, ``gap_az``, errores de localización,
        ``depto``, ``region``, ``estado``, ``portal`` y ``archivo``).
    """
    carpeta = Path(carpeta)
    rutas = sorted(carpeta.glob("reporte*.xlsx")) + sorted(carpeta.glob("Catalogo_Sismico*.csv"))
    if not rutas:
        raise FileNotFoundError(f"No hay archivos crudos del SGC en {carpeta}")
    partes = [(_leer_xlsx(r) if r.suffix == ".xlsx" else _leer_csv(r)) for r in rutas]
    return pd.concat(partes, ignore_index=True)


def limpiar(crudo: pd.DataFrame, m_min: float = cfg.M_MIN, inicio: str = cfg.INICIO) -> tuple:
    """Aplica, en orden y con trazabilidad, las reglas de limpieza del catálogo.

    Returns
    -------
    (catalogo, bitacora)
        ``catalogo``: sismos finales (columnas ``time, latitude, longitude, depth, mag, depto``).
        ``bitacora``: DataFrame con cada regla y el número de filas que eliminó.
    """
    pasos, df = [], crudo.copy()

    def paso(nombre, mascara_conservar):
        nonlocal df
        antes = len(df)
        df = df[mascara_conservar].copy()
        pasos.append({"regla": nombre, "filas_antes": antes, "eliminadas": antes - len(df), "filas_despues": len(df)})

    paso("solución revisada por analista (Revisado/manual; fuera 'automatic')", df.estado.isin(["revisado", "manual"]))
    paso("fecha, coordenadas y magnitud presentes", df[["time", "latitude", "longitude", "mag"]].notna().all(axis=1))
    paso("duplicados exactos (fecha-hora, lat, lon, magnitud)", ~df.duplicated(subset=["time", "latitude", "longitude", "mag"]))
    paso("sismo atribuido a un departamento colombiano (fuera mar y frontera)", df.depto.notna())
    paso(f"magnitud >= {m_min}", df.mag >= m_min)
    paso(f"desde {inicio}", df.time >= pd.Timestamp(inicio))
    df["depto"] = df["depto"].replace({"ARCHIPIELAGO DE SAN ANDRES. PROV. Y STA CATALINA": "SAN ANDRES"})
    df = df.sort_values("time").reset_index(drop=True)
    return df, pd.DataFrame(pasos)


def cargar_catalogo(reconstruir: bool = False, con_calidad: bool = False) -> pd.DataFrame:
    """Catálogo limpio (una fila por sismo).

    Parameters
    ----------
    reconstruir : bool
        Si es True rehace el ETL desde ``data/crudos`` y sobrescribe ``data/sismos_sgc_2004_2026.csv``.
    con_calidad : bool
        Si es True conserva las columnas de calidad de la localización (solo al reconstruir).
    """
    if reconstruir or not cfg.CATALOGO.exists():
        cat, _ = limpiar(leer_crudos())
        cat[["time", "latitude", "longitude", "depth", "mag", "depto"]].to_csv(cfg.CATALOGO, index=False)
        return cat if con_calidad else cat[["time", "latitude", "longitude", "depth", "mag", "depto"]]
    cat = pd.read_csv(cfg.CATALOGO, parse_dates=["time"])
    return cat


# ---------------------------------------------------------------------------
# 2. Indexación espacio-temporal
# ---------------------------------------------------------------------------
def indice_mes(fechas, inicio: str = cfg.INICIO) -> np.ndarray:
    """Número de mes desde ``inicio`` (enero de 2004 = 0)."""
    t0 = pd.Timestamp(inicio)
    f = pd.DatetimeIndex(fechas)
    return ((f.year - t0.year) * 12 + f.month - t0.month).to_numpy()


def fecha_de_mes(m) -> pd.DatetimeIndex:
    """Inverso de :func:`indice_mes` (primer día de cada mes)."""
    t0 = pd.Timestamp(cfg.INICIO)
    k = np.atleast_1d(np.asarray(m, dtype=int)) + (t0.month - 1)
    return pd.DatetimeIndex(pd.to_datetime({"year": t0.year + k // 12, "month": k % 12 + 1, "day": 1}))


def dimensiones_rejilla(celda: float = cfg.CELDA_GRADOS) -> tuple:
    """Número de filas y columnas de la rejilla de celdas sobre el rectángulo de Colombia.

    Returns
    -------
    tuple
        ``(n_lat, n_lon)``.
    """
    n_lat = int(np.ceil((cfg.LAT_MAX - cfg.LAT_MIN) / celda))
    n_lon = int(np.ceil((cfg.LON_MAX - cfg.LON_MIN) / celda))
    return n_lat, n_lon


def asignar_celda(lat, lon, celda: float = cfg.CELDA_GRADOS) -> np.ndarray:
    """Índice de celda (fila-mayor) de una rejilla regular en grados (EPSG:4326)."""
    n_lat, n_lon = dimensiones_rejilla(celda)
    i = np.clip(((np.asarray(lat) - cfg.LAT_MIN) / celda).astype(int), 0, n_lat - 1)
    j = np.clip(((np.asarray(lon) - cfg.LON_MIN) / celda).astype(int), 0, n_lon - 1)
    return i * n_lon + j


def centro_celda(c, celda: float = cfg.CELDA_GRADOS) -> tuple:
    """Coordenadas del centro de una o varias celdas.

    Parameters
    ----------
    c : int o array-like
        Identificador de celda (fila × n_lon + columna).

    Returns
    -------
    tuple
        ``(latitud, longitud)`` del centro.
    """
    n_lat, n_lon = dimensiones_rejilla(celda)
    c = np.asarray(c)
    return cfg.LAT_MIN + ((c // n_lon) + 0.5) * celda, cfg.LON_MIN + ((c % n_lon) + 0.5) * celda


def preparar_eventos(cat: pd.DataFrame, celda: float = cfg.CELDA_GRADOS) -> tuple:
    """Añade ``mes`` y ``celda`` y descarta el último mes si está incompleto.

    Returns
    -------
    (eventos, n_meses)
    """
    df = cat.copy()
    df["mes"] = indice_mes(df.time)
    ultimo = df.time.max()
    if (ultimo + pd.Timedelta(days=1)).month == ultimo.month:  # el último mes no ha terminado
        df = df[df.mes < df.mes.max()].copy()
    df["celda"] = asignar_celda(df.latitude, df.longitude, celda)
    return df.reset_index(drop=True), int(df.mes.max()) + 1


# ---------------------------------------------------------------------------
# 3. Panel celda × mes con variables causales (solo pasado)
# ---------------------------------------------------------------------------
def suma_movil(A: np.ndarray, k: int) -> np.ndarray:
    """Suma de los últimos ``k`` meses (incluido el actual) a lo largo del eje 1."""
    cs = np.cumsum(A, axis=1)
    out = cs.copy()
    out[:, k:] = cs[:, k:] - cs[:, :-k]
    return out


def vecindad(A: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    """Suma de las 8 celdas vecinas (rezago espacial tipo reina) para cada celda y mes."""
    n_m = A.shape[1]
    a = A.reshape(n_lat, n_lon, n_m)
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)))
    s = sum(p[1 + di:1 + di + n_lat, 1 + dj:1 + dj + n_lon] for di in (-1, 0, 1) for dj in (-1, 0, 1))
    return (s - a).reshape(n_lat * n_lon, n_m)


VARIABLES_BASE = ["n_1m", "n_3m", "n_6m", "n_12m", "n35_3m", "n35_12m", "mag_media_3m", "mag_media_12m",
                  "mag_max_3m", "mag_max_12m", "prof_media_hist", "vec_1m", "vec_3m", "n_mismo_mes_anio_prev",
                  "sen_mes", "cos_mes", "lat", "lon"]


def construir_panel(eventos: pd.DataFrame, n_meses: int, corte: str = cfg.CORTE_PRUEBA,
                    celda: float = cfg.CELDA_GRADOS, min_eventos: int = cfg.MIN_EVENTOS_CELDA) -> tuple:
    """Construye el panel de modelado.

    Cada fila es (celda activa ``c``, mes ``t``). Las variables usan solo sismos hasta el final del
    mes ``t``; el objetivo es el número de sismos M >= 2,5 de la celda en el mes ``t + 1``.
    Las celdas activas (>= ``min_eventos`` sismos) se eligen solo con meses anteriores al corte.

    Returns
    -------
    (X, info)
        ``X``: DataFrame con :data:`VARIABLES_BASE`.
        ``info``: diccionario con ``cc`` (celda), ``tt`` (mes de las variables), ``tm`` (mes objetivo),
        ``y`` (conteo objetivo), ``persist`` y ``estacional`` (líneas base), la matriz ``C`` de conteos
        celda × mes, ``m_test`` (índice del primer mes de prueba), ``celdas`` activas y la rejilla.
    """
    n_lat, n_lon = dimensiones_rejilla(celda)
    n_cel = n_lat * n_lon
    cell, m = eventos.celda.to_numpy(), eventos.mes.to_numpy()

    def denso(w):
        A = np.zeros((n_cel, n_meses))
        np.add.at(A, (cell, m), w)
        return A

    C = denso(np.ones(len(eventos)))
    C35 = denso((eventos.mag.to_numpy() >= 3.5).astype(float))
    SM = denso(eventos.mag.to_numpy())
    SD = denso(eventos.depth.clip(lower=0).to_numpy())
    MX = np.full((n_cel, n_meses), np.nan)
    g = eventos.groupby(["celda", "mes"]).mag.max()
    MX[g.index.get_level_values(0), g.index.get_level_values(1)] = g.to_numpy()

    f = {}
    for k in (1, 3, 6, 12):
        f[f"n_{k}m"] = suma_movil(C, k)
    for k in (3, 12):
        f[f"n35_{k}m"] = suma_movil(C35, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"mag_media_{k}m"] = np.where(f[f"n_{k}m"] > 0, suma_movil(SM, k) / np.maximum(f[f"n_{k}m"], 1), np.nan)
        f[f"mag_max_{k}m"] = pd.DataFrame(MX.T).rolling(k, min_periods=1).max().to_numpy().T
    cum_n = np.cumsum(C, axis=1)
    f["prof_media_hist"] = np.where(cum_n > 0, np.cumsum(SD, axis=1) / np.maximum(cum_n, 1), np.nan)
    for k in (1, 3):
        f[f"vec_{k}m"] = vecindad(suma_movil(C, k), n_lat, n_lon)

    m_test = int(indice_mes([corte])[0])
    celdas = np.where(C[:, :m_test].sum(1) >= min_eventos)[0]
    tt_all = np.arange(cfg.HISTORIA_MESES, n_meses - 1)
    cc, tt = np.meshgrid(celdas, tt_all, indexing="ij")
    cc, tt = cc.ravel(), tt.ravel()
    X = pd.DataFrame({k: v[cc, tt] for k, v in f.items()})
    X["n_mismo_mes_anio_prev"] = C[cc, tt - 11]
    mes_obj = (tt + 1) % 12
    X["sen_mes"] = np.sin(2 * np.pi * mes_obj / 12)
    X["cos_mes"] = np.cos(2 * np.pi * mes_obj / 12)
    X["lat"], X["lon"] = centro_celda(cc, celda)
    X = X[VARIABLES_BASE]
    info = dict(cc=cc, tt=tt, tm=tt + 1, y=C[cc, tt + 1], persist=C[cc, tt], estacional=C[cc, tt - 11],
                C=C, m_test=m_test, celdas=celdas, n_lat=n_lat, n_lon=n_lon, n_cel=n_cel, celda=celda,
                n_meses=n_meses)
    return X, info


def particion(info: dict, gap: int = cfg.GAP_MESES) -> tuple:
    """Índices de entrenamiento, gap y prueba (partición cronológica por mes objetivo)."""
    tm = info["tm"]
    tr = np.where(tm < info["m_test"] - gap)[0]
    gp = np.where((tm >= info["m_test"] - gap) & (tm < info["m_test"]))[0]
    te = np.where(tm >= info["m_test"])[0]
    return tr, gp, te


def pliegues_temporales(tm_tr, n_pliegues: int = cfg.N_PLIEGUES, gap: int = cfg.GAP_MESES,
                        min_frac: float = 0.4) -> list:
    """Validación cruzada temporal de ventana creciente con gap (análogo a ``TimeSeriesSplit``).

    Los meses de entrenamiento se dividen en ``n_pliegues`` bloques de validación consecutivos que
    cubren la parte final (``1 - min_frac``); cada pliegue entrena con todos los meses anteriores al
    bloque menos ``gap`` meses. Devuelve una lista de pares (idx_ajuste, idx_validación).
    """
    tm_tr = np.asarray(tm_tr)
    meses = np.unique(tm_tr)
    ini = int(len(meses) * min_frac)
    bloques = np.array_split(meses[ini:], n_pliegues)
    out = []
    for b in bloques:
        ajuste = np.where(tm_tr < b[0] - gap)[0]
        valid = np.where(np.isin(tm_tr, b))[0]
        out.append((ajuste, valid))
    return out


# ---------------------------------------------------------------------------
# 4. Variables de excitación tipo ETAS (ingeniería de características del Entregable 2)
# ---------------------------------------------------------------------------
class ExcitacionMensual:
    """Intensidad de excitación causal entre celdas inspirada en ETAS/Hawkes.

    ``S_c(t) = sum_{l>=0} rho^l sum_{c'} K_sigma(d_{cc'}) E_{c'}(t-l)`` con ``E`` la «energía relativa»
    ``sum 10^{alpha (M - M_MIN)}`` de los sismos de la celda ``c'`` en el mes y ``K_sigma`` un núcleo
    gaussiano con distancia en km. Solo usa sismos hasta el mes ``t``.
    """

    def __init__(self, eventos: pd.DataFrame, info: dict):
        self.ev, self.info = eventos, info
        src = np.where(info["C"].sum(1) > 0)[0]
        la_c, lo_c = centro_celda(info["celdas"], info["celda"])
        la_s, lo_s = centro_celda(src, info["celda"])
        dy = (la_c[:, None] - la_s[None, :]) * 111.0
        dx = (lo_c[:, None] - lo_s[None, :]) * 111.0 * np.cos(np.radians(la_c[:, None]))
        self.D2 = dx ** 2 + dy ** 2
        self.src = src
        fila = {c: i for i, c in enumerate(info["celdas"])}
        self.idx = np.array([fila[c] for c in info["cc"]])

    def energia(self, alpha: float) -> np.ndarray:
        """Productividad sísmica mensual por celda, ``Σ 10^(α (M − M_min))``, normalizada con los meses previos al corte."""
        info = self.info
        E = np.zeros((info["n_cel"], info["n_meses"]))
        np.add.at(E, (self.ev.celda.to_numpy(), self.ev.mes.to_numpy()), 10 ** (alpha * (self.ev.mag.to_numpy() - cfg.M_MIN)))
        E = E[self.src]
        return E / E[:, :info["m_test"]].sum(0).mean()     # normalización calculada solo con meses previos al corte

    @staticmethod
    def memoria(S0: np.ndarray, rho: float) -> np.ndarray:
        """Memoria exponencial de una serie mensual: ``S_t = S0_t + ρ S_{t−1}`` a lo largo de los meses."""
        return lfilter([1.0], [1.0, -rho], S0, axis=1)


def variables_excitacion(eventos: pd.DataFrame, info: dict) -> pd.DataFrame:
    """12 canales ``exc_a{alpha}_s{sigma}_r{rho}`` (log1p) y la media expansiva de la celda."""
    fis = ExcitacionMensual(eventos, info)
    F = {}
    for alpha in (0.5, 0.9):
        E = fis.energia(alpha)
        for sigma in (15.0, 30.0, 60.0):
            S0 = np.exp(-fis.D2 / (2 * sigma ** 2)) @ E
            for rho in (0.4, 0.8):
                S = fis.memoria(S0, rho)
                F[f"exc_a{alpha}_s{int(sigma)}_r{rho}"] = np.log1p(S[fis.idx, info["tt"]])
    cum = np.cumsum(info["C"], axis=1)
    F["media_expansiva"] = cum[info["cc"], info["tt"]] / (info["tt"] + 1.0)
    return pd.DataFrame(F)


def panel_completo(corte: str = cfg.CORTE_PRUEBA) -> tuple:
    """Atajo: catálogo -> eventos -> panel -> partición. Devuelve (eventos, X, info, tr, gap, te)."""
    ev, n_m = preparar_eventos(cargar_catalogo())
    X, info = construir_panel(ev, n_m, corte)
    tr, gp, te = particion(info)
    return ev, X, info, tr, gp, te


def panel_dataframe(X: pd.DataFrame, info: dict, idx=None) -> pd.DataFrame:
    """Panel en formato largo (variables + objetivo + identificadores) para el EDA.

    Columnas añadidas: ``celda``, ``mes_obj`` (índice), ``fecha_obj`` (primer día del mes objetivo), ``y`` y
    ``log_y`` = log(1 + y).
    """
    idx = np.arange(len(X)) if idx is None else np.asarray(idx)
    df = X.iloc[idx].reset_index(drop=True).copy()
    df["celda"] = info["cc"][idx]
    df["mes_obj"] = info["tm"][idx]
    df["fecha_obj"] = fecha_de_mes(info["tm"][idx])
    df["y"] = info["y"][idx]
    df["log_y"] = np.log1p(df["y"])
    return df
