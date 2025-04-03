import gymnasium as gym
import keyboard
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Crear el entorno con el modo de renderizado "rgb_array"
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

obs, info = env.reset()

done = False

# Diccionario para mapear las teclas a las acciones del juego
action_mapping = {
    'up': 1,     # Mover hacia arriba
    'down': 4,   # Mover hacia abajo
    'left': 3,   # Mover hacia la izquierda
    'right': 2   # Mover hacia la derecha
}

# Definir el orden de los canales
channel_order = ['ghost_alive', 'ghost_dead', 'pacman', 'block', 'big_block', 'food']

# Configurar la ventana de matplotlib
plt.ion()  # Habilitar modo interactivo
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
img_original = axs[0, 0].imshow(np.zeros((172, 160, 3)))  # Frame original

# Imágenes para los 6 canales
channel_imgs = [ax.imshow(np.zeros((172, 160)), cmap='gray') for ax in axs.ravel()[1:]]

# Cargar las plantillas
templates = {
    'ghost_left': cv2.imread('imgs/g3.png', 0),
    'ghost_right': cv2.imread('imgs/g4.png', 0),
    'ghost_dead_left': cv2.imread('imgs/dg1.png', 0),
    'ghost_dead_right': cv2.imread('imgs/dg2.png', 0),
    'block': cv2.imread('imgs/b1.png', 0),
    'big_block': cv2.imread('imgs/sb2.png', 0),
    'cherry': cv2.imread('imgs/c1.png', 0),
    'strawberry': cv2.imread('imgs/strawberry2.png', 0),
    'orange': cv2.imread('imgs/orange2.png', 0),
    'pretzel': cv2.imread('imgs/pretzel2.png', 0),
    'apple': cv2.imread('imgs/apple2.png', 0),
    'pear': cv2.imread('imgs/pera2.png', 0),
    'banana': cv2.imread('imgs/banana1.png', 0),
    'pacman_up': cv2.imread('imgs/FinalPacman/Up_final_template.png', 0),
    'pacman_down': cv2.imread('imgs/FinalPacman/Down_final_template.png', 0),
    'pacman_left': cv2.imread('imgs/FinalPacman/Left_final_template.png', 0),
    'pacman_right': cv2.imread('imgs/FinalPacman/Right_final_template.png', 0),
    'pacman_left2': cv2.imread('imgs/FinalPacman/PacManLeft3.png', 0),
    'pacman_right2': cv2.imread('imgs/FinalPacman/PacManRight3.png', 0)
}

# Colores para cada tipo de objeto (en formato BGR)
colors = {
    'ghost_left': (0, 0, 255),  # Rojo para fantasmas mirando a la izquierda
    'ghost_right': (255, 0, 0),  # Azul para fantasmas mirando a la derecha
    'ghost_dead_left': (128, 0, 128),  # Púrpura para fantasmas muertos mirando a la izquierda
    'ghost_dead_right': (128, 0, 128),  # Púrpura para fantasmas muertos mirando a la derecha
    'block': (0, 255, 0),   # Verde para bloques pequeños
    'big_block': (0, 255, 255),  # Amarillo para bloques grandes
    'cherry': (0, 165, 255),  # Naranja para cerezas
    'strawberry': (255, 165, 0),  # Naranja para fresa
    'orange': (255, 165, 0),  # Naranja para naranja
    'pretzel': (255, 165, 0),  # Naranja para pretzel
    'apple': (255, 165, 0),  # Naranja para manzana
    'pear': (255, 165, 0),  # Naranja para pera
    'banana': (255, 165, 0),  # Naranja para banana
    'pacman_up': (255, 255, 0),  # Amarillo para Pac-Man mirando hacia arriba
    'pacman_down': (255, 255, 0),  # Amarillo para Pac-Man mirando hacia abajo
    'pacman_left': (255, 255, 0),  # Amarillo para Pac-Man mirando hacia la izquierda
    'pacman_right': (255, 255, 0),  # Amarillo para Pac-Man mirando hacia la derecha
    'pacman_left2': (255, 255, 0),  # Amarillo para Pac-Man mirando hacia la izquierda
    'pacman_right2': (255, 255, 0)  # Amarillo para Pac-Man mirando hacia la derecha
}

# Diccionario para controlar qué objetos detectar (1 para detectar, 0 para ignorar)
detect_objects = {
    'ghost_left': 1,
    'ghost_right': 1,
    'ghost_dead_left': 1,
    'ghost_dead_right': 1,  #Fantastas ------------------------
    'block': 1,
    'big_block': 1, #Bloques ------------------------
    'cherry': 1,
    'strawberry': 1,
    'orange': 1,
    'pretzel': 1,
    'apple': 1,
    'pear': 1,
    'banana': 1, #Alimentos ------------------------
    'pacman_up': 1,
    'pacman_down': 1,
    'pacman_left': 1,
    'pacman_right': 1,
    'pacman_left2': 1,
    'pacman_right2': 1 #Pacman ------------------------
}

chanels = {'ghost_alive':['ghost_left', 'ghost_right'], 
           'ghost_dead':['ghost_dead_left', 'ghost_dead_right'], 
           'pacman':['pacman_up', 'pacman_down', 'pacman_left', 'pacman_right', 'pacman_left2', 'pacman_right2'],
           'block':['block'],
           'big_block':['big_block'],
           'food':['cherry', 'strawberry', 'orange', 'pretzel', 'apple', 'pear', 'banana']}


detection_array = np.zeros((len(templates), 210, 160), dtype=int)  # Ajustar las dimensiones según el frame

# Definir el orden de los canales
channel_order = ['ghost_alive', 'ghost_dead', 'pacman', 'block', 'big_block', 'food']

while not done:
    action = None

    # Captura la entrada del teclado
    if keyboard.is_pressed('up'):
        action = action_mapping['up']
    elif keyboard.is_pressed('down'):
        action = action_mapping['down']
    elif keyboard.is_pressed('left'):
        action = action_mapping['left']
    elif keyboard.is_pressed('right'):
        action = action_mapping['right']
    elif keyboard.is_pressed('esc'):
        done = True
    
    # Si se presionó alguna tecla válida, ejecutar la acción
    if action is not None:
        obs, reward, done, truncated, info = env.step(action)

    # Renderizar el entorno capturando el frame
    frame = env.render()
    frame = frame[:172, :, :]  # Recortar el frame

    # Convertir el frame a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Crear un array de 6 canales para almacenar las detecciones
    detection_channels = np.zeros((172, 160, 6), dtype=np.uint8)

    # Crear máscaras individuales para cada tipo de objeto
    object_masks = {obj: np.zeros((172, 160), dtype=np.uint8) for obj in templates.keys()}

    for template_name, template in templates.items():
        if detect_objects[template_name]:
            res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(object_masks[template_name], pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), 255, -1)

    # Combinar las máscaras por categoría y aplicarlas al frame en escala de grises
    for i, category in enumerate(channel_order):
        category_mask = np.zeros((172, 160), dtype=np.uint8)
        for obj in chanels[category]:
            category_mask = cv2.bitwise_or(category_mask, object_masks[obj])
        detection_channels[:,:,i] = cv2.bitwise_and(frame_gray, category_mask)

    # Actualizar las imágenes en matplotlib
    img_original.set_data(frame)
    axs[0, 0].set_title("Frame Original")

    # Actualizar los 6 canales
    for i, (img, category) in enumerate(zip(channel_imgs, channel_order)):
        img.set_data(detection_channels[:,:,i])
        axs[(i+1)//4, (i+1)%4].set_title(category)

    # Forzar la actualización de los límites de las imágenes
    for img in [img_original] + channel_imgs:
        img.autoscale()

    # Redibujar la figura completa
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.001)  # Pausar brevemente para permitir la actualización de la ventana

env.close()
plt.ioff()  # Deshabilitar modo interactivo
plt.show()
