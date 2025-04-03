import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keyboard

# Crear directorio para guardar capturas (plantillas)
if not os.path.exists("plantillas"):
    os.makedirs("plantillas")

# Crear el entorno clásico de MsPacman
env = gym.make("MsPacman-v4")
obs, info = env.reset()

# Dimensiones de destino: 128x128
target_dim = (128, 128)

# Convertir la primera observación a escala de grises y redimensionar
gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
frame_resized = cv2.resize(gray_obs, target_dim, interpolation=cv2.INTER_NEAREST)

# Variable global para guardar el frame actual (en 128x128 y en escala de grises)
current_frame = frame_resized.copy()

# Configurar la figura de Matplotlib
fig, ax = plt.subplots()
img = ax.imshow(frame_resized, cmap='gray')
ax.set_title("MsPacman 128x128 (Grises) - Click para capturar, ESC para salir")

# Función callback para capturar el screenshot al hacer clic
def on_click(event):
    global current_frame
    # Al hacer clic, se guarda el frame actual (ya en 128x128 y grises)
    filename = os.path.join("plantillas", f"template_{int(time.time()*1000)}.png")
    cv2.imwrite(filename, current_frame)
    print(f"Captura guardada: {filename}")

# Conectar el evento de click a la figura
fig.canvas.mpl_connect('button_press_event', on_click)

# Mapeo de teclas para controlar el juego (según Gym clásico: 0 = NOOP, 1 = Up, 2 = Right, 3 = Left, 4 = Down)
action_mapping = {
    'up': 1,
    'right': 2,
    'left': 3,
    'down': 4
}

running = True
current_action = 0

plt.show(block=False)

while running:
    # Comprobar si se presionó alguna tecla direccional para mover a Pacman
    action = 0  # NOOP por defecto
    for direction, action_num in action_mapping.items():
        if keyboard.is_pressed(direction):
            action = action_num
            break

    # Ejecutar un paso en el entorno
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        obs, info = env.reset()

    # Convertir a escala de grises y redimensionar a 128x128
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    frame_resized = cv2.resize(gray_obs, target_dim, interpolation=cv2.INTER_NEAREST)
    current_frame = frame_resized.copy()  # Actualizar el frame actual

    # Actualizar la imagen en la figura
    img.set_data(frame_resized)
    fig.canvas.draw_idle()

    plt.pause(0.01)

    # Salir si se presiona ESC
    if keyboard.is_pressed('esc'):
        running = False

env.close()
plt.close()
print("¡Programa finalizado!")