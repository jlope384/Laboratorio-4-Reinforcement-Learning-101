# Laboratorio #4 y #5 — RL, Gymnasium y ALE (Space Invaders)

CC3092 - Deep Learning y Sistemas Inteligentes

Repositorio: https://github.com/jlope384/Laboratorio-4-Reinforcement-Learning-101

## Contenido del repositorio

### Laboratorio 4 — Fundamentos de RL y Gymnasium

- `notebook/Laboratorio4_RL_Gymnasium.ipynb` — notebook completo: investigación de fundamentos de RL, investigación
  de Gymnasium, módulo exploratorio (agente aleatorio en CartPole-v1 y FrozenLake-v1, política simple para
  CartPole-v1) y gráficas generadas.
- `reporte/Reporte_Laboratorio4.pdf` — reporte escrito (2 páginas) con la investigación resumida, resultados y
  discusión.
- `build_notebook.py` / `build_report.py` — scripts usados para generar el notebook y el reporte PDF.

### Laboratorio 5 — ALE y Space Invaders

- `ale_module.py` — módulo reutilizable para interactuar con ALE: `crear_entorno`, `agente_aleatorio`,
  `agente_regla_simple`, `ejecutar_episodio`, `generar_video_agente`. Funciona tanto para entornos de Atari como
  para cualquier otro entorno de Gymnasium, y está pensado como base para laboratorios/proyecto futuros.
- `notebook/Laboratorio5_ALE_SpaceInvaders.ipynb` — notebook completo: investigación de ALE y sus espacios de
  observación/acción, uso del módulo y resultados (video + métricas) de un agente aleatorio y uno de regla simple
  jugando `ALE/SpaceInvaders-v5`.
- `notebook/videos/` — videos `.mp4` generados por `generar_video_agente`.
- `reporte/Reporte_Laboratorio5.pdf` — reporte escrito (2 páginas) con la investigación y la descripción del módulo.
- `build_notebook_lab5.py` / `build_report_lab5.py` — scripts usados para generar el notebook y el reporte PDF.

### Común

- `docs/` — enunciados de ambos laboratorios.
- `requirements.txt` — dependencias de Python.

## Cómo ejecutar

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt jupyter
python -m AutoROM --accept-license   # descarga las ROMs de Atari (una sola vez, para el Lab 5)

jupyter notebook notebook/Laboratorio4_RL_Gymnasium.ipynb
jupyter notebook notebook/Laboratorio5_ALE_SpaceInvaders.ipynb
```

Se recomienda Python 3.10–3.12 (Gymnasium aún no tiene soporte estable en 3.14+).
