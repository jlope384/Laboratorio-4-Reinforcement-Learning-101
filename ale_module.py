"""Modulo reutilizable para interactuar con el Arcade Learning Environment (ALE) via Gymnasium.

Funciones:
    crear_entorno        -- crea un entorno de Gymnasium (Atari/ALE o cualquier otro), con
                             grabacion de video opcional.
    agente_aleatorio     -- agente baseline que muestrea acciones al azar.
    agente_regla_simple  -- agente basado en una regla fija (sin aprendizaje), ciclo
                             disparar / moverse.
    ejecutar_episodio    -- corre un episodio completo con la funcion de agente dada.
    generar_video_agente -- crea el entorno con grabacion, corre n episodios y devuelve
                             las rutas de los videos generados junto con las metricas.

Este modulo se disenio para ser la base de infraestructura de labs/proyectos futuros que
entrenen agentes sobre entornos de Atari (Space Invaders y otros).
"""
import glob
import os

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)


def crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, name_prefix="rl-video", **make_kwargs):
    """Crea y retorna un entorno de Gymnasium.

    Funciona tanto para entornos de Atari/ALE (ej. "ALE/SpaceInvaders-v5") como para
    cualquier otro entorno de Gymnasium (ej. "CartPole-v1"). Si se especifica
    ``video_folder``, el entorno se envuelve con ``gymnasium.wrappers.RecordVideo`` para
    grabar los episodios indicados por ``episode_trigger`` (por defecto, todos).
    """
    if video_folder is not None:
        make_kwargs["render_mode"] = "rgb_array"

    env = gym.make(nombre_entorno, **make_kwargs)

    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
        )

    return env


def agente_aleatorio(observation, env):
    """Agente baseline: retorna una accion muestreada al azar de env.action_space."""
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """Agente basado en una regla fija (sin entrenamiento): dispara y se desplaza en un
    ciclo constante (FIRE, RIGHT, FIRE, LEFT), aprovechando que en Space Invaders casi
    siempre conviene estar disparando mientras se recorre el eje horizontal.

    Guarda su contador de paso como atributo del propio entorno para no depender de
    estado externo a la funcion.
    """
    ciclo = [1, 2, 1, 3]  # indices de accion: FIRE, RIGHT, FIRE, LEFT
    contador = getattr(env, "_regla_simple_contador", 0)
    accion = ciclo[contador % len(ciclo)] if env.action_space.n > max(ciclo) else env.action_space.sample()
    env._regla_simple_contador = contador + 1
    return accion


def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None):
    """Ejecuta un episodio completo usando funcion_agente hasta terminated/truncated o
    max_steps. Retorna (steps, total_reward).
    """
    observation, info = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    total_reward = 0.0

    while not (terminated or truncated) and steps < max_steps:
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward

    return steps, total_reward


def generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1, seed=0, max_steps=10000, **make_kwargs):
    """Crea el entorno con grabacion de video habilitada, ejecuta n_episodios episodios
    completos con funcion_agente, cierra el entorno (indispensable para que el video se
    escriba a disco) y retorna (rutas_de_video, metricas), donde metricas es una lista de
    dicts con steps y return por episodio.
    """
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix=name_prefix,
        **make_kwargs,
    )

    metricas = []
    for ep in range(n_episodios):
        steps, total_reward = ejecutar_episodio(env, funcion_agente, max_steps=max_steps, seed=seed + ep)
        metricas.append({"episodio": ep + 1, "steps": steps, "return": total_reward})

    env.close()

    videos = sorted(glob.glob(os.path.join(video_folder, f"{name_prefix}*.mp4")))
    return videos, metricas
