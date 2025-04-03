import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keyboard

# ============================
#       CONFIGURACIÓN
# ============================

# Lista de nombres de plantillas (fantasmas)
GHOST_TEMPLATES = ["ghost_1.png", "ghost_2.png", "ghost_3.png", "ghost_4.png", "ghost_5.png", "ghost_6.png", "ghost_7.png", "ghost_8.png", "ghost_9.png", "ghost_10.png", "ghost_11.png", "ghost_12.png", "ghost_13.png", "ghost_14.png", "ghost_15.png"]

# Crear directorio para guardar capturas (plantillas), si no existe
if not os.path.exists("plantillas"):
    os.makedirs("plantillas")

# Crear el entorno de MsPacman
env = gym.make("MsPacman-v4")
obs, info = env.reset()

# Dimensiones de destino: 128x128
target_dim = (128, 128)

# Convertir la primera observación a escala de grises y redimensionar
gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
frame_resized = cv2.resize(gray_obs, target_dim, interpolation=cv2.INTER_NEAREST)

# ============================
#    CARGA DE PLANTILLAS
# ============================
templates = []
for ghost_file in GHOST_TEMPLATES:
    template_path = os.path.join("plantillas", ghost_file)
    ghost_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if ghost_template is None:
        print(f"No se pudo cargar la plantilla: {template_path}")
    else:
        templates.append(ghost_template)

if not templates:
    print("No se cargaron plantillas válidas. Saliendo...")
    env.close()
    exit()

# Obtenemos dimensiones de cada plantilla y las guardamos para usarlas en el dibujo
templates_info = []
for tpl in templates:
    w, h = tpl.shape[::-1]  # w=ancho, h=alto
    templates_info.append((tpl, w, h))

# ============================
#    FIGURA DE MATPLOTLIB
# ============================
# Creamos una figura con 2 subplots:
# - ax1 (izquierda): MsPacman en gris
# - ax2 (derecha)  : MsPacman en color con la detección resaltada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Mostramos el primer frame en ambos subplots
img1 = ax1.imshow(frame_resized, cmap='gray')
ax1.set_title("MsPacman 128x128 (Grises)")

# Para la detección, convertimos el frame a color (para dibujar rectángulos)
frame_resized_color = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)
img2 = ax2.imshow(frame_resized_color)
ax2.set_title("Detección de fantasmas")

plt.tight_layout()

# ============================
#     EVENTO DE CAPTURA
# ============================
# Variable para guardar el frame actual en gris (128x128)
current_frame = frame_resized.copy()

def on_click(event):
    """Al hacer clic en cualquier parte de la figura,
       se guarda el frame actual (128x128 y gris) en la carpeta plantillas."""
    global current_frame
    filename = os.path.join("plantillas", f"template_{int(time.time()*1000)}.png")
    cv2.imwrite(filename, current_frame)
    print(f"Captura guardada: {filename}")

# Conectar el evento de click a la figura
fig.canvas.mpl_connect('button_press_event', on_click)

# ============================
#   MAPEO DE TECLAS (Gym)
# ============================
# Gym clásico: 0 = NOOP, 1 = Up, 2 = Right, 3 = Left, 4 = Down
action_mapping = {
    'up': 1,
    'right': 2,
    'left': 3,
    'down': 4
}

running = True

plt.show(block=False)

# ============================
#       BUCLE PRINCIPAL
# ============================
while running:
    # Comprobar si se presionó alguna tecla direccional
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
    current_frame = frame_resized.copy()  # Guardamos la versión gris actual

    # ============================
    #  DETECCIÓN DE FANTASMAS
    # ============================
    # Para dibujar la detección en color, convertimos el frame a BGR
    frame_resized_color = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

    # Umbral para considerar detecciones
    threshold = 0.8

    # Recorremos todas las plantillas
    for idx, (tpl, w, h) in enumerate(templates_info):
        # Aplicamos la correlación normalizada
        result = cv2.matchTemplate(frame_resized, tpl, cv2.TM_CCOEFF_NORMED)

        # Buscamos el valor máximo y su ubicación
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Si supera el umbral, dibujamos un rectángulo
        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # Dibujamos un rectángulo de color distinto para cada plantilla
            # Ejemplo: alternar entre rojo, verde y azul
            colors = [(0,0,255), (0,255,0), (255,0,0)]
            cv2.rectangle(frame_resized_color, top_left, bottom_right, colors[idx % len(colors)], 2)

    # ============================
    #   ACTUALIZAR FIGURA
    # ============================
    # Subplot izquierda (gris)
    img1.set_data(frame_resized)
    # Subplot derecha (color con detección)
    img2.set_data(frame_resized_color)

    # Redibujamos la figura
    fig.canvas.draw_idle()
    plt.pause(0.01)

    # Salir si se presiona ESC
    if keyboard.is_pressed('esc'):
        running = False

env.close()
plt.close()
print("¡Programa finalizado!")