"""Genera el notebook del Laboratorio 5 (ALE y Space Invaders) con nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
# Portada
# ---------------------------------------------------------------------------
md("""# Laboratorio #5 — Agentes en el Arcade Learning Environment (ALE): Space Invaders

**Curso:** CC3092 - Deep Learning y Sistemas Inteligentes

**Contenido:**
1. El Arcade Learning Environment (ALE)
2. Espacios de observación y acción en entornos Atari
3. Módulo de desarrollo: funciones para interactuar con ALE
""")

# ---------------------------------------------------------------------------
# SECCIÓN 1: El ALE
# ---------------------------------------------------------------------------
md("## 1. El Arcade Learning Environment (ALE)")

md("""### 1.1 ¿Qué es el ALE y qué problema resuelve? Relación con Stella y con Atari 2600

El Arcade Learning Environment (ALE) es una capa de software que expone los juegos originales de la Atari 2600
como entornos de RL con una interfaz programática: recibe acciones (los mismos botones del joystick original) y
devuelve el frame de video, el puntaje y si el juego terminó. El problema que resuelve es que antes de ALE, para
usar un juego de Atari como benchmark de un agente había que meterse a mano con el emulador, leer memoria RAM a
ciegas y programar toda la lógica de "qué es un episodio" y "cuál es la recompensa" por separado para cada juego.
ALE estandariza todo eso: mismo tipo de entrada/salida para cualquiera de los ~60 juegos que soporta, lo que
permitió que Atari se convirtiera en el benchmark clásico de RL profundo (empezando con el paper de DQN de
DeepMind en 2013/2015).

Por dentro, ALE no reimplementa la consola: usa **Stella**, un emulador de Atari 2600 de código abierto, como
motor de emulación real. ALE se monta encima de Stella y añade la capa de "entorno de RL": extrae el frame
renderizado, lee la RAM del sistema, calcula la recompensa a partir del puntaje del juego, y decide cuándo termina
un episodio (game over). Los "juegos originales de Atari 2600" son, literalmente, las ROMs binarias de los
cartuchos originales; ALE (vía Stella) las ejecuta tal cual corrían en el hardware de 1977, ciclo a ciclo.
""")

md("""### 1.2 Diferencias entre variantes de un mismo juego en ALE

Un mismo juego en ALE puede exponerse de varias formas, todas ejecutando la misma ROM pero cambiando qué se
observa o cómo se procesan las acciones:

- **`ALE/SpaceInvaders-v5`** (sin sufijo) es la variante "estándar": observación en imagen RGB, con los valores
  por defecto recomendados de frame skipping y sticky actions que se describen abajo.
- El sufijo **`-ram`** (`ALE/SpaceInvaders-ram-v5`) cambia la observación: en vez de la imagen RGB, entrega
  directamente los 128 bytes de memoria RAM de la consola como vector de observación. Es el mismo juego y la misma
  dinámica, solo cambia qué "ve" el agente.
- **`frameskip`** controla cuántos frames del emulador se repiten (bajo el capó) por cada `step()` que pide el
  agente; por defecto en la v5 el propio ALE aplica un frame skip aleatorio entre 2 y 4 frames. Se puede fijar a
  un entero para que sea determinístico, o a 1 para desactivarlo y recibir control frame por frame.
- **`repeat_action_probability`** (sticky actions) es la probabilidad de que, en vez de ejecutar la acción que
  el agente acaba de elegir, el emulador repita la acción anterior. En la v5 viene activado por defecto en 0.25.
  Se agregó porque el emulador es determinístico: sin esto, un agente podría memorizar la secuencia exacta de
  acciones que gana el juego sin aprender realmente una política robusta; con sticky actions ya no puede confiar
  en que su acción se va a ejecutar siempre, así que tiene que generalizar.
- **`full_action_space`** decide si el espacio de acción expone las 18 acciones posibles del joystick de Atari (el
  set completo, incluyendo combinaciones que ese juego en particular ignora) o solo el subconjunto mínimo de
  acciones que de verdad tienen efecto en ese juego. Por defecto está en `False`, así que Space Invaders expone
  solo sus 6 acciones relevantes en vez de las 18.
""")

md("""### 1.3 Frame skipping: por qué importa

Los juegos de Atari corren a 60 frames por segundo, pero de un frame al siguiente casi no cambia nada: un agente
que decidiera una acción nueva en cada uno de esos 60 frames por segundo estaría gastando cómputo carísimo
(renderizar + que el agente evalúe su política) para una diferencia de información mínima entre frames
consecutivos. El frame skipping resuelve eso: el agente elige una acción y esa misma acción se mantiene durante
varios frames seguidos (típicamente 4) antes de pedirle la siguiente decisión.

Esto afecta la velocidad de simulación de forma directa, porque reduce en ese mismo factor cuántas veces hay que
correr la política del agente por segundo de juego simulado, lo cual importa muchísimo cuando se entrena con
millones de pasos. Y afecta el aprendizaje porque acorta el horizonte temporal efectivo del problema: en vez de
tener que aprender a tomar una decisión distinta en cada uno de 60 frames por segundo, el agente aprende sobre una
secuencia de decisiones más corta y con cambios de estado más perceptibles entre un paso y el siguiente, lo que en
la práctica hace el crédito temporal (qué acción causó qué recompensa) más fácil de aprender.
""")

md("""### 1.4 Space Invaders: mecánica, objetivo y recompensa

Space Invaders pone al jugador a controlar un cañón que se mueve horizontalmente en la parte baja de la pantalla,
disparando hacia una formación de invasores alienígenas que avanza en bloque de lado a lado y va bajando cada vez
que toca un borde. El objetivo es destruir a todos los invasores antes de que la formación llegue hasta abajo (lo
que termina el juego) o antes de que el jugador pierda todas sus vidas por los disparos de los aliens; también hay
naves bonus ("mystery ship") que cruzan arriba y dan puntos extra. Cada tipo de invasor vale una cantidad de
puntos distinta (los de las filas de atrás valen más, porque son más difíciles de alcanzar según la formación va
bajando), y matarlos todos hace aparecer una formación nueva, más rápida.

En el entorno de ALE, la señal de recompensa se deriva directamente del puntaje del juego original: cada vez que
el agente destruye un invasor (o la nave bonus), la RAM del juego incrementa el score interno, y ALE reporta como
`reward` el incremento de puntaje ocurrido en ese paso (por defecto sin normalizar ni recortar, aunque wrappers
como `AtariPreprocessing` sí pueden aplicar un recorte de recompensa). No hay una recompensa "por sobrevivir" como
en CartPole: aquí la señal es completamente dependiente de que el agente logre impactar a los invasores.
""")

# ---------------------------------------------------------------------------
# SECCIÓN 2: Espacios de observación y acción
# ---------------------------------------------------------------------------
md("## 2. Espacios de observación y acción en entornos Atari")

md("""### 2.1 Espacio de observación por defecto y comparación con CartPole-v1

El espacio de observación por defecto de `ALE/SpaceInvaders-v5` es `Box(0, 255, (210, 160, 3), uint8)`: una imagen
RGB de 210 píxeles de alto por 160 de ancho con 3 canales de color, es decir el frame de video tal cual lo
renderiza el emulador. Son 100,800 valores por observación, cada uno un entero entre 0 y 255.

En el laboratorio anterior, `CartPole-v1` usaba `Box(4,)`: un vector de solo 4 números reales (posición y
velocidad del carro, ángulo y velocidad angular del poste), ya "resumidos" en cantidades físicas directamente
interpretables. La diferencia es enorme: en CartPole el estado relevante ya viene extraído y compacto, mientras
que en Space Invaders el agente recibe el píxel crudo y tiene que aprender por sí mismo a extraer de ahí qué es
"mi nave", "los invasores" y "las balas". Eso implica que hace falta algún tipo de procesamiento de imagen (en la
práctica, una red convolucional) para convertir esos píxeles en una representación útil, y que el espacio de
entrada es varios órdenes de magnitud más grande y más redundante (frames consecutivos comparten casi toda la
imagen) que el vector de bajo nivel de CartPole. También implica mucho más costo de memoria y cómputo por cada
observación, lo cual es justo la razón por la que existen wrappers como `AtariPreprocessing` (ver 2.4).
""")

md("""### 2.2 Observación en RAM (`obs_type="ram"`)

La variante `obs_type="ram"` (equivalente a usar el id `ALE/SpaceInvaders-ram-v5`) expone directamente los 128
bytes de memoria de la Atari 2600 como observación, en vez de la imagen renderizada: un vector `Box(0, 255, (128,),
uint8)`. Esa RAM es donde el juego guarda su estado interno completo (posición del jugador, posición y patrón de
los invasores, puntaje, vidas, etc.), así que en principio contiene toda la información necesaria para jugar, solo
que codificada de forma opaca y específica de cada juego (hay que saber qué byte representa qué para poder leerla
con sentido).

Se preferiría la variante RAM sobre la de imagen sobre todo cuando se quiere reducir muchísimo el costo
computacional: 128 valores es una fracción minúscula comparado con los 100,800 de la imagen, así que entrenar
sobre RAM es mucho más rápido y no requiere convoluciones, solo una red densa. También es útil cuando se quiere
hacer ingeniería de features a mano (identificar manualmente qué bytes corresponden a qué) o para depurar/analizar
el estado interno del juego. La desventaja es que la interpretación de esos bytes no es estándar entre juegos y no
se generaliza a un agente que deba aprender "a partir de lo que se ve", que es el escenario más realista y el que
más se estudia en RL profundo con Atari.
""")

md("""### 2.3 Espacio de acción de Space Invaders

Con la configuración por defecto (`full_action_space=False`), Space Invaders expone `Discrete(6)`, es decir 6
acciones posibles (verificado también en el código de la sección 3):

- `NOOP` (0): no hacer nada, dejar el cañón quieto sin disparar.
- `FIRE` (1): disparar sin moverse.
- `RIGHT` (2): mover el cañón hacia la derecha.
- `LEFT` (3): mover el cañón hacia la izquierda.
- `RIGHTFIRE` (4): moverse a la derecha y disparar en el mismo paso.
- `LEFTFIRE` (5): moverse a la izquierda y disparar en el mismo paso.

Estas seis son las únicas acciones que de verdad tienen efecto distinto en este juego (el joystick de Atari en
general soporta más combinaciones, como diagonales, que no aplican a un juego que solo se mueve en un eje). Si se
activara `full_action_space=True` se expondrían las 18 acciones del joystick completo, aunque para Space Invaders
las adicionales serían redundantes con estas 6.
""")

md("""### 2.4 `AtariPreprocessing` y `FrameStackObservation`

`AtariPreprocessing` es un wrapper de Gymnasium pensado específicamente para entornos de Atari, que aplica en
conjunto varias transformaciones que se volvieron estándar desde el paper original de DQN:

- Convierte la imagen a **escala de grises** (de RGB a un solo canal), porque el color casi no aporta información
  útil para jugar y así se reduce el tamaño de la observación a un tercio.
- **Redimensiona** el frame a 84x84 píxeles, un tamaño mucho más manejable para una CNN que los 210x160 originales.
- Aplica **frame skipping** internamente (repitiendo la acción elegida durante varios frames y tomando el máximo
  entre los últimos dos, para evitar parpadeos por sprites que Atari dibuja en frames alternados).
- Opcionalmente aplica **recorte de recompensa** (reward clipping), típicamente a {-1, 0, +1}, para que la escala
  de recompensa sea comparable entre juegos con puntajes muy distintos y no desestabilice el entrenamiento.

`FrameStackObservation` (antes `FrameStack`) apila las últimas *k* observaciones consecutivas (por ejemplo, 4
frames en escala de grises de 84x84) en un solo tensor. Se usa junto con `AtariPreprocessing` porque un solo frame
estático no contiene información de movimiento: viendo un único frame no se puede saber hacia dónde se dirige una
bala o si la formación de invasores se está moviendo a la izquierda o a la derecha. Apilar varios frames
consecutivos le da al agente (o a la red) la información temporal mínima para inferir velocidades y direcciones,
sin tener que usar una arquitectura recurrente.
""")

# ---------------------------------------------------------------------------
# SECCIÓN 3: Módulo de desarrollo
# ---------------------------------------------------------------------------
md("## 3. Módulo de desarrollo: funciones para interactuar con ALE")

md("""Las funciones reutilizables se implementaron en un módulo aparte, `ale_module.py`, en la raíz del
repositorio (no directamente en el notebook), para que puedan importarse desde aquí o desde cualquier laboratorio
o proyecto futuro sin copiar y pegar código. El módulo expone:

- **`crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, name_prefix="rl-video", **make_kwargs)`**
  crea y retorna un entorno de Gymnasium con `gym.make`. Si se especifica `video_folder`, fuerza
  `render_mode="rgb_array"` y envuelve el entorno con `gymnasium.wrappers.RecordVideo`, grabando los episodios que
  indique `episode_trigger` (por defecto, todos). Funciona igual para `ALE/SpaceInvaders-v5` que para cualquier
  otro entorno de Gymnasium (CartPole, FrozenLake, etc.), porque no asume nada específico de Atari.
- **`agente_aleatorio(observation, env)`** retorna `env.action_space.sample()`: el baseline sin entrenamiento.
- **`agente_regla_simple(observation, env)`** es un agente basado en una regla fija (tampoco aprendida): recorre
  un ciclo constante `FIRE, RIGHT, FIRE, LEFT`, aprovechando que en Space Invaders casi siempre conviene estar
  disparando mientras se recorre el eje horizontal para cubrir toda la formación de invasores.
- **`ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None)`** corre un episodio completo con la
  función de agente dada hasta que `terminated` o `truncated` sea verdadero (o se llegue a `max_steps`), y retorna
  `(steps, total_reward)`.
- **`generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1, seed=0,
  max_steps=10000, **make_kwargs)`** combina las anteriores: crea el entorno con grabación habilitada, corre
  `n_episodios` episodios completos, cierra el entorno con `env.close()` (indispensable para que
  `RecordVideo` termine de escribir el `.mp4` a disco) y retorna las rutas de los videos generados junto con la
  lista de métricas (steps y return) de cada episodio.
""")

md("### 3.1 Importar el módulo y verificar el entorno de Space Invaders")
code("""from ale_module import (
    crear_entorno,
    agente_aleatorio,
    agente_regla_simple,
    ejecutar_episodio,
    generar_video_agente,
)

env = crear_entorno("ALE/SpaceInvaders-v5")
print("=== ALE/SpaceInvaders-v5 ===")
print("Espacio de observacion:", env.observation_space)
print("Espacio de accion:", env.action_space)
print("Significado de las acciones:", env.unwrapped.get_action_meanings())
env.close()
""")

md("""### 3.2 Video y métricas: agente aleatorio en Space Invaders

Se genera 1 episodio completo con `agente_aleatorio`, grabando el video con `generar_video_agente`.""")
code("""videos_random, metricas_random = generar_video_agente(
    "ALE/SpaceInvaders-v5",
    agente_aleatorio,
    video_folder="videos",
    name_prefix="space-invaders-random",
    n_episodios=1,
    seed=0,
)

print("Videos generados:", videos_random)
for m in metricas_random:
    print(f"Episodio {m['episodio']}: steps = {m['steps']:4d} | return = {m['return']:.1f}")
""")

md("""### 3.3 Video y métricas: agente de regla simple en Space Invaders

Como comparación adicional (no requerida como mínimo, pero incluida en el módulo), se genera también 1 episodio
con `agente_regla_simple` para contrastar contra el agente aleatorio.""")
code("""videos_regla, metricas_regla = generar_video_agente(
    "ALE/SpaceInvaders-v5",
    agente_regla_simple,
    video_folder="videos",
    name_prefix="space-invaders-regla-simple",
    n_episodios=1,
    seed=0,
)

print("Videos generados:", videos_regla)
for m in metricas_regla:
    print(f"Episodio {m['episodio']}: steps = {m['steps']:4d} | return = {m['return']:.1f}")
""")

md("""### 3.4 Comparación de resultados""")
code("""print("Resumen comparativo (1 episodio cada uno, Space Invaders):")
print(f"  Agente aleatorio -> steps: {metricas_random[0]['steps']:4d} | return: {metricas_random[0]['return']:.1f}")
print(f"  Regla simple     -> steps: {metricas_regla[0]['steps']:4d} | return: {metricas_regla[0]['return']:.1f}")
""")

md("""El agente de regla simple (disparar constantemente mientras se recorre el eje horizontal) obtuvo un return
mayor que el agente puramente aleatorio, lo cual tiene sentido: en Space Invaders la recompensa depende
exclusivamente de impactar invasores, y `agente_aleatorio` reparte su probabilidad entre las 6 acciones por igual,
así que solo dispara (`FIRE`, `RIGHTFIRE` o `LEFTFIRE`) la mitad de las veces y buena parte de esos disparos
ocurren sin que el cañón se haya movido a una posición útil. La regla simple, en cambio, siempre está disparando y
alterna el movimiento, lo que aumenta la probabilidad de que cada disparo tenga oportunidad de impactar algo.
Ninguno de los dos agentes fue entrenado: la diferencia viene enteramente de la heurística incorporada, igual que
se observó con CartPole-v1 en el laboratorio anterior.""")

nb["cells"] = cells

with open("notebook/Laboratorio5_ALE_SpaceInvaders.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generado en notebook/Laboratorio5_ALE_SpaceInvaders.ipynb")
