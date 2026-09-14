# Laboratorio #4 — Fundamentos de RL y Gymnasium

CC3092 - Deep Learning y Sistemas Inteligentes

Repositorio: https://github.com/jlope384/Laboratorio-4-Reinforcement-Learning-101

## Contenido del repositorio

- `notebook/Laboratorio4_RL_Gymnasium.ipynb` — notebook completo: investigación de fundamentos de RL, investigación
  de Gymnasium, módulo exploratorio (agente aleatorio en CartPole-v1 y FrozenLake-v1, política simple para
  CartPole-v1) y gráficas generadas.
- `reporte/Reporte_Laboratorio4.pdf` — reporte escrito (2 páginas) con la investigación resumida, resultados y
  discusión.
- `docs/` — enunciado del laboratorio.
- `requirements.txt` — dependencias de Python.
- `build_notebook.py` / `build_report.py` — scripts usados para generar el notebook y el reporte PDF.

## Cómo ejecutar

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt jupyter
jupyter notebook notebook/Laboratorio4_RL_Gymnasium.ipynb
```

Se recomienda Python 3.10–3.12 (Gymnasium aún no tiene soporte estable en 3.14+).
