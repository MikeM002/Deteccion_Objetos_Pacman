import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Si deseas forzar un backend (opcional)
# import matplotlib
# matplotlib.use('TkAgg')

# Crear el entorno clásico de MsPacman
env = gym.make("MsPacman-v4")
obs, info = env.reset()

# Configurar la figura para visualizar el frame
fig, ax = plt.subplots()
img = ax.imshow(obs)
ax.set_title("MsPacman - Usa las flechas para mover y ESC para salir")

# Diccionario para mapear las teclas a las acciones
# Según Gym clásico: 0 = NOOP, 1 = Up, 2 = Right, 3 = Left, 4 = Down
action_mapping = {
    'up': 1,
    'right': 2,
    'left': 3,
    'down': 4
}

# Variable global para almacenar la acción actual (por defecto NOOP: 0)
current_action = 0
running = True

# Función para capturar eventos de teclado
def on_key(event):
    global current_action, running
    key = event.key.lower()  # Convertir a minúsculas para uniformidad
    if key in action_mapping:
        current_action = action_mapping[key]
    elif key == 'escape':
        running = False

# Conectar el evento de teclado a la figura
fig.canvas.mpl_connect('key_press_event', on_key)

# Bucle principal para interactuar con el entorno
while running:
    # Realiza un paso en el entorno usando la acción actual
    obs, reward, terminated, truncated, info = env.step(current_action)
    done = terminated or truncated

    # Actualiza la imagen del frame
    img.set_data(obs)
    fig.canvas.draw_idle()

    # Si el episodio terminó, reinicia el entorno
    if done:
        obs, info = env.reset()

    # Reinicia la acción a NOOP para esperar la siguiente pulsación
    current_action = 0

    plt.pause(0.02)  # Pausa breve para actualizar la visualización

env.close()
plt.ioff()
plt.show()
print("¡MsPacman probado y detenido!")
