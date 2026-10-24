"""Experimentos complementarios del Entregable 2 (capítulos 14, 15, 16 y 20). Cada uno guarda sus
resultados en resultados/e2/ y se salta si ya existen.

  semillas_opt   cap. 14  estabilidad de los 4 optimizadores ante 5 semillas (XGBoost regresión)
  multifidelidad cap. 15  Random Search vs Successive Halving vs Hyperband (XGBoost regresión)
  computo        cap. 16  complejidad empírica y versiones optimizadas (KNN/FAISS, SAGA, partial_fit,
                          XGBoost hist/exact/early stopping, SVM, joblib, perfilamiento, memoria)
  semillas_mod   cap. 20  sensibilidad de los mejores modelos a la semilla de inicialización

Uso:  python scripts/experimentos_e2.py [nombre ...]      (sin nombres: todos)
"""
import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import benchmark as bm
from src import config as cfg
from src import datos
from src import evaluacion as ev
from src import modelos as mdl

OUT = cfg.RESULTADOS / "e2"
OUT.mkdir(parents=True, exist_ok=True)
_cache = {}


def base():
    if not _cache:
        cfg.fijar_semillas()
        ev_, X, info, tr, gp, te = datos.panel_completo()
        _cache.update(X=X, info=info, tr=tr, te=te, y=info["y"].astype(float), tm=info["tm"])
    return _cache


def matrices():
    """Predictoras preprocesadas (Pipeline del cap. 10 ajustado solo con entrenamiento) como arreglos densos."""
    b = base()
    if "Xtr" not in b:
        prep = mdl.preprocesador(b["X"].columns, "constante").fit(b["X"].iloc[b["tr"]])
        b["Xtr"] = np.asarray(prep.transform(b["X"].iloc[b["tr"]]), dtype=np.float32)
        b["Xte"] = np.asarray(prep.transform(b["X"].iloc[b["te"]]), dtype=np.float32)
    return b


def cronometrar(f, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r


def memoria_pico(f):
    tracemalloc.start(); r = f(); _, pico = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return pico / 2 ** 20, r


# ---------------------------------------------------------------------------
def semillas_opt():
    ruta = OUT / "semillas_optimizadores.csv"
    if ruta.exists():
        return
    b = base(); X, y, tm, tr = b["X"], b["y"], b["tm"], b["tr"]
    internos = bm.pliegues_internos(tm[tr], 3)

    def f(params):
        r = []
        for a, v in internos:
            m = bm.construir("reg", "XGBoost", params, X.columns, n_jobs=2).fit(X.iloc[tr[a]], y[tr[a]])
            r.append(-ev.rmsle(y[tr[v]], m.predict(X.iloc[tr[v]])))
        return float(np.mean(r))
    filas = []
    for o in bm.OPTIMIZADORES:
        for s in (1, 2, 3, 4, 5):
            t0 = time.perf_counter()
            params, val, reg = bm.Optimizador(o, f, "reg", "XGBoost", 15, s).correr()
            for t, (v, ti) in enumerate(zip(reg.valores, reg.tiempos)):
                filas.append({"optimizador": o, "semilla": s, "evaluacion": t + 1, "RMSLE": -v, "t_eval_s": ti})
            print(f"  {o} semilla {s}: {-val:.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


# ---------------------------------------------------------------------------
def multifidelidad():
    ruta = OUT / "multifidelidad.csv"
    if ruta.exists():
        return
    b = base(); X, y, tm, tr = b["X"], b["y"], b["tm"], b["tr"]
    meses = np.unique(tm[tr])
    ajuste = tr[tm[tr] < meses[-12] - cfg.GAP_MESES]          # selección sin el último año del entrenamiento
    reserva = tr[tm[tr] >= meses[-12]]                         # último año: evaluación de la configuración elegida
    internos = bm.pliegues_internos(tm[ajuste], 3)
    R = 600                                                    # recurso máximo: número de árboles
    esp = [p for p in bm.ESPACIOS[("reg", "XGBoost")] if p[0] != "n_estimators"]
    rng = np.random.default_rng(cfg.SEMILLA)

    def muestra():
        u = rng.random(len(esp))
        out = {}
        for p, ui in zip(esp, u):
            lo, hi = p[2], p[3]
            v = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo)))) if p[4] else lo + ui * (hi - lo)
            out[p[0]] = int(round(v)) if p[1] == "int" else v
        return out

    filas = []

    def evaluar(metodo, ronda, cfg_id, params, r):
        t0 = time.perf_counter()
        v = []
        for a, w in internos:
            m = bm.construir("reg", "XGBoost", {**params, "n_estimators": int(r)}, X.columns, n_jobs=2).fit(X.iloc[ajuste[a]], y[ajuste[a]])
            v.append(ev.rmsle(y[ajuste[w]], m.predict(X.iloc[ajuste[w]])))
        filas.append({"metodo": metodo, "ronda": ronda, "config": cfg_id, "recurso": int(r), "RMSLE_cv": float(np.mean(v)),
                      "t_s": time.perf_counter() - t0, "costo_equiv": r / R, **params})
        return float(np.mean(v))

    def halving(metodo, n, r0, eta=3, offset=0):
        confs = [(offset + i, muestra()) for i in range(n)]
        r, ronda = r0, 0
        while True:
            res = sorted(((evaluar(metodo, ronda, i, c, r), i, c) for i, c in confs), key=lambda t: t[0])
            if r >= R or len(res) == 1:
                return res[0]
            confs = [(i, c) for _, i, c in res[: max(1, len(res) // eta)]]
            r, ronda = min(R, r * eta), ronda + 1

    # Random Search: 15 configuraciones con el recurso completo (presupuesto = 15 equivalentes)
    for i in range(15):
        evaluar("Random Search", 0, i, muestra(), R)
    # Successive Halving: 27 configs a R/9, 9 a R/3, 3 a R (presupuesto = 9 equivalentes)
    halving("Successive Halving", 27, R / 9)
    # Hyperband (η = 3, s_max = 2): brackets s = 2, 1, 0 (presupuesto ≈ 14,7 equivalentes)
    halving("Hyperband", 27, R / 9, offset=0)
    halving("Hyperband", 5, R / 3, offset=100)
    halving("Hyperband", 3, R, offset=200)
    df = pd.DataFrame(filas)
    # evaluación de la mejor configuración (recurso completo) de cada método en el año reservado
    final = []
    for metodo, g in df[df.recurso == R].groupby("metodo"):
        mejor = g.sort_values("RMSLE_cv").iloc[0]
        params = {p[0]: (int(mejor[p[0]]) if p[1] == "int" else float(mejor[p[0]])) for p in esp}
        m = bm.construir("reg", "XGBoost", {**params, "n_estimators": R}, X.columns, n_jobs=2).fit(X.iloc[ajuste], y[ajuste])
        final.append({"metodo": metodo, "RMSLE_cv_mejor": mejor.RMSLE_cv, "RMSLE_reserva": ev.rmsle(y[reserva], m.predict(X.iloc[reserva])),
                      "evaluaciones": int((df.metodo == metodo).sum()), "costo_equiv": float(df[df.metodo == metodo].costo_equiv.sum()),
                      "tiempo_s": float(df[df.metodo == metodo].t_s.sum()), **params})
    df.to_csv(ruta, index=False)
    pd.DataFrame(final).to_csv(OUT / "multifidelidad_resumen.csv", index=False)


# ---------------------------------------------------------------------------
def computo():
    if (OUT / "computo_resumen.csv").exists():
        return
    import faiss
    from joblib import Parallel, delayed
    from memory_profiler import memory_usage
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDClassifier, SGDRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import NearestNeighbors
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    b = matrices()
    Xtr, Xte = b["Xtr"], b["Xte"]
    ytr, yte = np.log1p(b["y"][b["tr"]]), b["y"][b["te"]]
    btr, bte = (b["y"][b["tr"]] >= 1).astype(int), (b["y"][b["te"]] >= 1).astype(int)
    rng = np.random.default_rng(cfg.SEMILLA)
    N = len(Xtr)
    tamanos = [n for n in (2000, 4000, 8000, 16000, N)]
    q = Xte[: 5000]
    filas, escal = [], []

    def reg_fila(familia, version, complejidad, t_fit, t_pred, mem, metrica, valor, extra=None):
        filas.append({"familia": familia, "version": version, "complejidad": complejidad, "t_ajuste_s": t_fit,
                      "t_inferencia_s": t_pred, "memoria_MiB": mem, "metrica": metrica, "valor": valor, **(extra or {})})

    # --- KNN: exhaustiva vs KD-Tree vs Ball Tree vs FAISS exacto/aproximado -----------------------------
    k = 75
    exacto = {}
    for n in tamanos:
        idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
        A = np.ascontiguousarray(Xtr[idx])
        for alg in ("brute", "kd_tree", "ball_tree"):
            nn = NearestNeighbors(n_neighbors=k, algorithm=alg)
            t_fit, _ = cronometrar(lambda: nn.fit(A))
            t_q, (_, I) = cronometrar(lambda: nn.kneighbors(q), rep=2)
            escal.append({"familia": "KNN", "version": alg, "n": n, "t_ajuste_s": t_fit, "t_inferencia_s": t_q})
            if alg == "brute":
                exacto[n] = I
            if n == N:
                p = np.expm1(ytr[I].mean(1))
                mem, _ = memoria_pico(lambda: NearestNeighbors(n_neighbors=k, algorithm=alg).fit(A).kneighbors(q[:1000]))
                reg_fila("KNN", alg, {"brute": "O(n·p) por consulta", "kd_tree": "O(log n) por consulta (p bajo)", "ball_tree": "O(log n) por consulta"}[alg],
                         t_fit, t_q, mem, "RMSLE (5000 consultas)", ev.rmsle(yte[:5000], p), {"recall@k": 1.0})
        for nombre, construir_idx in (("FAISS exacto (FlatL2)", lambda: faiss.IndexFlatL2(A.shape[1])),
                                      ("FAISS aproximado (IVF, nprobe=4)", None)):
            def armar():
                if construir_idx is not None:
                    ix = construir_idx(); ix.add(A); return ix
                nlist = max(8, int(np.sqrt(n)))
                quant = faiss.IndexFlatL2(A.shape[1]); ix = faiss.IndexIVFFlat(quant, A.shape[1], nlist)
                ix.train(A); ix.add(A); ix.nprobe = 4; return ix
            t_fit, ix = cronometrar(armar)
            t_q, (_, I) = cronometrar(lambda: ix.search(np.ascontiguousarray(q), k), rep=2)
            rec = float(np.mean([len(set(a) & set(bb)) / k for a, bb in zip(I, exacto[n])]))
            escal.append({"familia": "KNN", "version": nombre, "n": n, "t_ajuste_s": t_fit, "t_inferencia_s": t_q, "recall@k": rec})
            if n == N:
                p = np.expm1(ytr[np.clip(I, 0, None)].mean(1))
                reg_fila("KNN", nombre, "O(n·p) por consulta (vectorizado)" if construir_idx else "O((n/nlist)·nprobe·p) por consulta",
                         t_fit, t_q, np.nan, "RMSLE (5000 consultas)", ev.rmsle(yte[:5000], p), {"recall@k": rec})

    # --- Ridge / Lasso: solver directo vs SAGA / SGD ------------------------------------------------------
    for nombre, f, cplx in (("Ridge · cholesky", lambda: Ridge(alpha=1.0, solver="cholesky"), "O(n·p² + p³)"),
                            ("Ridge · SAGA", lambda: Ridge(alpha=1.0, solver="saga", max_iter=2000, tol=1e-4, random_state=cfg.SEMILLA), "O(n·p) por época"),
                            ("Lasso · descenso coordenado", lambda: Lasso(alpha=1e-4, max_iter=5000), "O(n·p) por ciclo"),
                            ("Lasso · SGD (L1)", lambda: SGDRegressor(penalty="l1", alpha=1e-4, max_iter=50, random_state=cfg.SEMILLA), "O(n·p) por época")):
        for n in tamanos:
            idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
            t_fit, m = cronometrar(lambda: f().fit(Xtr[idx], ytr[idx]))
            escal.append({"familia": nombre.split(" ·")[0], "version": nombre, "n": n, "t_ajuste_s": t_fit})
        mem, m = memoria_pico(lambda: f().fit(Xtr, ytr))
        t_pred, p = cronometrar(lambda: m.predict(Xte))
        reg_fila(nombre.split(" ·")[0], nombre, cplx, t_fit, t_pred, mem, "RMSLE prueba", ev.rmsle(yte, np.expm1(p)))

    # --- Naive Bayes: fit completo vs partial_fit por lotes ------------------------------------------------
    t_fit, nb = cronometrar(lambda: GaussianNB().fit(Xtr, btr))
    mem, _ = memoria_pico(lambda: GaussianNB().fit(Xtr, btr))
    reg_fila("Naive Bayes", "fit (todo en memoria)", "O(n·p)", t_fit, cronometrar(lambda: nb.predict_proba(Xte))[0], mem,
             "AUC prueba", roc_auc_score(bte, nb.predict_proba(Xte)[:, 1]))

    def incremental():
        m = GaussianNB()
        for i in range(0, N, 2000):
            m.partial_fit(Xtr[i:i + 2000], btr[i:i + 2000], classes=[0, 1])
        return m
    t_fit, nb2 = cronometrar(incremental)
    mem, _ = memoria_pico(lambda: GaussianNB().partial_fit(Xtr[:2000], btr[:2000], classes=[0, 1]))
    reg_fila("Naive Bayes", "partial_fit (lotes de 2000)", "O(n·p), memoria O(lote·p)", t_fit, cronometrar(lambda: nb2.predict_proba(Xte))[0], mem,
             "AUC prueba", roc_auc_score(bte, nb2.predict_proba(Xte)[:, 1]))

    # --- XGBoost: exact vs hist, early stopping --------------------------------------------------------------
    par = dict(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=cfg.SEMILLA, n_jobs=2, verbosity=0)
    for tm_ in ("exact", "hist"):
        for n in tamanos:
            idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
            t_fit, _ = cronometrar(lambda: XGBRegressor(tree_method=tm_, **par).fit(Xtr[idx], ytr[idx]), rep=2)
            escal.append({"familia": "XGBoost", "version": f"XGBoost · {tm_}", "n": n, "t_ajuste_s": t_fit})
        mem, m = memoria_pico(lambda: XGBRegressor(tree_method=tm_, **par).fit(Xtr, ytr))
        reg_fila("XGBoost", f"XGBoost · {tm_} (600 árboles)", "O(K·d·n·p·log n)" if tm_ == "exact" else "O(K·d·(n·p + B·p)), B = 256 bins",
                 t_fit, cronometrar(lambda: m.predict(Xte))[0], mem, "RMSLE prueba", ev.rmsle(yte, np.expm1(m.predict(Xte))), {"arboles": 600})
    tmtr = b["tm"][b["tr"]]
    ult = tmtr >= np.unique(tmtr)[-12]
    t_fit, m = cronometrar(lambda: XGBRegressor(tree_method="hist", early_stopping_rounds=30, **{**par, "n_estimators": 2000})
                           .fit(Xtr[~ult], ytr[~ult], eval_set=[(Xtr[ult], ytr[ult])], verbose=False), rep=2)
    reg_fila("XGBoost", "XGBoost · hist + early stopping", "O(K*·d·(n·p + B·p)), K* elegido en validación", t_fit,
             cronometrar(lambda: m.predict(Xte))[0], np.nan, "RMSLE prueba", ev.rmsle(yte, np.expm1(m.predict(Xte))), {"arboles": int(m.best_iteration) + 1})

    # --- SVM: núcleo RBF vs LinearSVC vs SGD --------------------------------------------------------------
    for n in (1000, 2000, 4000, 8000):
        idx = rng.choice(N, n, replace=False)
        t_fit, _ = cronometrar(lambda: SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr[idx], btr[idx]), rep=2)
        escal.append({"familia": "SVM", "version": "SVC RBF", "n": n, "t_ajuste_s": t_fit})
    for n in tamanos:
        idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
        for nombre, f in (("LinearSVC", lambda: LinearSVC(C=0.1, dual=False)), ("SGD (hinge)", lambda: SGDClassifier(loss="hinge", alpha=1e-4, random_state=cfg.SEMILLA))):
            t_fit, _ = cronometrar(lambda: f().fit(Xtr[idx], btr[idx]))
            escal.append({"familia": "SVM", "version": nombre, "n": n, "t_ajuste_s": t_fit})
    idx = rng.choice(N, 8000, replace=False)
    for nombre, f, X_, y_, cplx in (("SVC RBF (8000 filas)", lambda: SVC(kernel="rbf", C=1.0, gamma="scale"), Xtr[idx], btr[idx], "O(n²·p) a O(n³)"),
                                    ("LinearSVC (todas las filas)", lambda: LinearSVC(C=0.1, dual=False), Xtr, btr, "O(n·p) por iteración"),
                                    ("SGD hinge (todas las filas)", lambda: SGDClassifier(loss="hinge", alpha=1e-4, random_state=cfg.SEMILLA), Xtr, btr, "O(n·p) por época")):
        t_fit, m = cronometrar(lambda: f().fit(X_, y_), rep=1 if "RBF" in nombre else 3)
        mem, _ = memoria_pico(lambda: f().fit(X_, y_))
        reg_fila("SVM", nombre, cplx, t_fit, cronometrar(lambda: m.decision_function(Xte), rep=1)[0], mem, "AUC prueba", roc_auc_score(bte, m.decision_function(Xte)))

    # --- joblib: secuencial vs loky vs threading ----------------------------------------------------------
    pl = bm.pliegues_internos(tmtr, 3)
    tareas = [(a, v, s) for a, v in pl for s in range(4)]           # 12 ajustes de árbol (3 pliegues × 4 semillas)

    def ajuste(a, v, s):
        m = DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=s).fit(Xtr[a], ytr[a])
        return float(np.sqrt(np.mean((m.predict(Xtr[v]) - ytr[v]) ** 2)))
    par_res = []
    for nombre, kw in (("secuencial", dict(n_jobs=1)), ("loky (2 procesos)", dict(n_jobs=2, backend="loky")),
                       ("threading (2 hilos)", dict(n_jobs=2, backend="threading"))):
        t, _ = cronometrar(lambda: Parallel(**kw)(delayed(ajuste)(a, v, s) for a, v, s in tareas))
        par_res.append({"estrategia": nombre, "tiempo_s": t})
    par_df = pd.DataFrame(par_res)
    par_df["aceleracion"] = par_df.tiempo_s.iloc[0] / par_df.tiempo_s
    par_df.to_csv(OUT / "computo_joblib.csv", index=False)

    # --- perfilamiento de una evaluación del bucle interno -------------------------------------------------
    X, y, tm, tr = b["X"], b["y"], b["tm"], b["tr"]
    internos = bm.pliegues_internos(tm[tr], 3)
    params = {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.5}

    def una_eval():
        from sklearn.metrics import roc_auc_score as auc
        r = []
        for a, v in internos:
            m = bm.construir("clf", "Random Forest", params, X.columns, "SMOTE", n_jobs=1).fit(X.iloc[tr[a]], (y[tr[a]] >= 1).astype(int))
            r.append(auc((y[tr[v]] >= 1).astype(int), m.predict_proba(X.iloc[tr[v]])[:, 1]))
        return r
    pr = cProfile.Profile(); pr.enable(); una_eval(); pr.disable()
    s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("cumulative")
    filas_perf = []
    for (fn, ln, nombre), (cc, nc, tt, ct, _) in st.stats.items():
        filas_perf.append({"funcion": f"{Path(fn).name}:{ln}({nombre})", "llamadas": nc, "t_propio_s": tt, "t_acumulado_s": ct})
    pd.DataFrame(filas_perf).sort_values("t_propio_s", ascending=False).head(25).to_csv(OUT / "computo_perfil.csv", index=False)
    memoria_rss()
    pd.DataFrame(escal).to_csv(OUT / "computo_escalamiento.csv", index=False)
    pd.DataFrame(filas).to_csv(OUT / "computo_resumen.csv", index=False)


def memoria_rss():
    """Incremento de memoria residente (RSS, incluye asignaciones en C/C++) de cada proceso, en un subproceso limpio."""
    import subprocess
    codigo = r'''
import sys, json, numpy as np
sys.path.insert(0, %r)
from memory_profiler import memory_usage
from scripts.experimentos_e2 import matrices, base
from src import benchmark as bm, config as cfg
b = matrices(); Xtr, Xte = b["Xtr"], b["Xte"]; ytr = np.log1p(b["y"][b["tr"]]); btr = (b["y"][b["tr"]] >= 1).astype(int)
nombre = sys.argv[1]
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
idx = np.random.default_rng(cfg.SEMILLA).choice(len(Xtr), 8000, replace=False)
f = {"Random Forest (150 árboles)": lambda: RandomForestRegressor(150, max_depth=10, min_samples_leaf=5, max_features=0.5, max_samples=0.5, n_jobs=1).fit(Xtr, ytr),
     "XGBoost exact (600 árboles)": lambda: XGBRegressor(tree_method="exact", n_estimators=600, max_depth=4, n_jobs=2).fit(Xtr, ytr),
     "XGBoost hist (600 árboles)": lambda: XGBRegressor(tree_method="hist", n_estimators=600, max_depth=4, n_jobs=2).fit(Xtr, ytr),
     "KNN brute (k=75, consulta de la prueba)": lambda: NearestNeighbors(n_neighbors=75, algorithm="brute").fit(Xtr).kneighbors(Xte),
     "KNN kd_tree (k=75, consulta de la prueba)": lambda: NearestNeighbors(n_neighbors=75, algorithm="kd_tree").fit(Xtr).kneighbors(Xte),
     "SVC RBF (8000 filas)": lambda: SVC().fit(Xtr[idx], btr[idx])}[nombre]
m = memory_usage((f, (), {}), interval=0.01)
print(json.dumps({"proceso": nombre, "rss_base_MiB": m[0], "incremento_pico_MiB": max(m) - m[0]}))
''' % str(Path(__file__).resolve().parents[1])
    filas = []
    for nombre in ("Random Forest (150 árboles)", "XGBoost exact (600 árboles)", "XGBoost hist (600 árboles)",
                   "KNN brute (k=75, consulta de la prueba)", "KNN kd_tree (k=75, consulta de la prueba)", "SVC RBF (8000 filas)"):
        out = subprocess.run([sys.executable, "-c", codigo, nombre], capture_output=True, text=True)
        filas.append(json.loads(out.stdout.strip().splitlines()[-1]))
    pd.DataFrame(filas).to_csv(OUT / "computo_memoria.csv", index=False)


# ---------------------------------------------------------------------------
def semillas_mod():
    ruta = OUT / "semillas_modelos.csv"
    if ruta.exists():
        return
    from sklearn.metrics import roc_auc_score
    b = base(); X, y, tr, te = b["X"], b["y"], b["tr"], b["te"]
    m = pd.read_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv")
    m = m[m.estado == "completado"]
    reg = m[m.tarea == "reg"].sort_values("externo_media").groupby("modelo").head(1)
    reg = reg[reg.modelo.isin(["XGBoost", "Random Forest", "Árbol de decisión", "KNN"])]
    clf = m[m.tarea == "clf"].sort_values("externo_media", ascending=False).groupby("modelo").head(1)
    clf = clf[clf.modelo.isin(["XGBoost", "Random Forest", "Árbol de decisión", "SVM"])]
    filas = []
    for _, f in pd.concat([reg, clf]).iterrows():
        params = json.loads(f.params_finales)
        for s in (1, 2, 3, 4, 5, 42):
            t0 = time.perf_counter()
            if f.tarea == "reg":
                mod = bm.construir("reg", f.modelo, params, X.columns, semilla=s, n_jobs=2).fit(X.iloc[tr], y[tr])
                val = ev.rmsle(y[te], mod.predict(X.iloc[te])); met = "RMSLE"
            else:
                mod = bm.construir("clf", f.modelo, params, X.columns, f.balanceo, semilla=s, n_jobs=2).fit(X.iloc[tr], (y[tr] >= 1).astype(int))
                val = roc_auc_score((y[te] >= 1).astype(int), bm.puntaje_clf(mod, X.iloc[te])); met = "ROC_AUC"
            filas.append({"tarea": f.tarea, "modelo": f.modelo, "balanceo": f.balanceo, "optimizador": f.optimizador, "semilla": s,
                          "metrica": met, "valor": val, "t_s": time.perf_counter() - t0})
            print(f"  {f.tarea} {f.modelo} semilla {s}: {val:.4f}", flush=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


if __name__ == "__main__":
    todos = {"memoria": memoria_rss, "semillas_opt": semillas_opt, "multifidelidad": multifidelidad, "computo": computo, "semillas_mod": semillas_mod}
    for nombre in (sys.argv[1:] or list(todos)):
        t0 = time.time()
        print(f"== {nombre}", flush=True)
        todos[nombre]()
        print(f"== {nombre} listo en {(time.time() - t0) / 60:.1f} min", flush=True)
