# Despliegue del dashboard en Render (paso a paso)

Pasos para publicar el dashboard del Entregable 3 con un enlace público.

## 1. Subir el repositorio a GitHub

1. En GitHub Desktop: **File → Add local repository** y elige la carpeta del proyecto (`sismos-colombia-e3`).
   Si dice que no es un repositorio Git, acepta **create a repository** aquí mismo.
2. Haz commit con el mensaje `Entregable 3: dashboard y libro` (botón **Commit to main**).
3. Pulsa **Publish repository**. Nombre sugerido: `sismos-colombia-e3`. Deja **Keep this code private** desmarcada:
   el repositorio público permite que el profesor vea el código. Render también funciona con repositorios privados
   si le das acceso, pero el público es más simple.
   No agregues ningún `origin` a mano: GitHub Desktop lo crea solo.

La carpeta incluye `artifacts/` (modelos serializados, pocos MB) y `data/`. No incluye `resultados/origen_movil/matrices/`
(unos 50 MB; se regeneran con `scripts/origen_movil_final.py --matrices`) ni `book/_build/`.

## 2. Crear el servicio en Render

1. Entra a https://render.com con la cuenta de GitHub (**Sign in with GitHub**).
2. **New → Blueprint**. Busca el repositorio `sismos-colombia-e3` y conéctalo (si no aparece, usa **Configure account**
   para dar acceso al repositorio).
3. Render lee `render.yaml`: servicio web `sismos-colombia-dash`, plan **Free**, build `pip install -r requirements.txt`,
   start `gunicorn app:server ...` y la variable `PYTHON_VERSION = 3.9.25`. Revisa que aparezcan y pulsa **Apply**.
   Si prefieres sin Blueprint: **New → Web Service**, mismos valores y la variable `PYTHON_VERSION` en **Environment**.
4. Espera el build (5 a 15 minutos la primera vez; `shap` y `numba` tardan). Al final verás **Live**.
5. La URL pública aparece arriba, con la forma `https://sismos-colombia-dash.onrender.com`. Si el nombre ya estaba
   tomado, Render le agrega un sufijo: copia la que aparezca.

## 3. Comprobar

- Abre la URL. Deben cargar las páginas **Contexto**, **EDA** y **Modelos**.
- En **Modelos**, prueba una predicción en vivo y cambia de modelo.
- Si el build falla, copia el error del registro y pásamelo: casi siempre es una versión de librería.

## 4. Avisos del plan gratuito

- El servicio se duerme tras unos 15 minutos sin visitas. El primer acceso tarda cerca de un minuto en despertar.
  Abre la URL 5 minutos antes de la sustentación.
- Respaldo sin internet: `dashboard_sismos_E3.html`.

## 5. Después de tener la URL

Pásame la URL. Reemplazo `<servicio>` en el capítulo 0, el capítulo 7 y el README, y regenero el libro.
