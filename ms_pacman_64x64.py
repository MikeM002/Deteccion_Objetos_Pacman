import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Crear el entorno clásico de MsPacman
env = gym.make("MsPacman-v4")
obs, info = env.reset()

# Dimensiones a las que se redimensionará el frame (64x64)
resize_dim = (64, 64)

# Configurar la figura para visualizar el frame redimensionado
fig, ax = plt.subplots()
# Redimensionar la primera observación
obs_resized = cv2.resize(obs, resize_dim, interpolation=cv2.INTER_NEAREST)
img = ax.imshow(obs_resized)
ax.set_title("MsPacman (64x64) - Usa las flechas para mover y ESC para salir")

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
    obs, reward, terminated, truncated, info = env.step(current_action)
    done = terminated or truncated

    # Si el episodio terminó, reinicia el entorno
    if done:
        obs, info = env.reset()

    # Redimensionar el frame a 64x64
    obs_resized = cv2.resize(obs, resize_dim, interpolation=cv2.INTER_NEAREST)
    # Actualizar la imagen mostrada
    img.set_data(obs_resized)
    fig.canvas.draw_idle()

    # Reinicia la acción a NOOP para esperar la siguiente pulsación
    current_action = 0
    plt.pause(0.02)  # Pequeña pausa para actualizar la visualización

env.close()
plt.ioff()
plt.show()
print("¡MsPacman 64x64 probado y detenido!")
