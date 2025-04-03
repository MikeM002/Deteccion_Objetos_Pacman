import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2
import keyboard

class Cortador:
    def __init__(self):
        self.env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
        self.obs, _ = self.env.reset()
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self.img1 = self.ax1.imshow(self.env.render())
        self.img2 = self.ax2.imshow(np.zeros_like(self.obs))
        
        self.rect_selector = RectangleSelector(
            self.ax1, self.on_select, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
        self.crop = None
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.action_mapping = {
            'up': 1,
            'down': 4,
            'left': 3,
            'right': 2
        }
        
        plt.show(block=False)

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop = self.obs[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        self.img2.set_data(self.crop)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 's':
            if self.crop is not None:
                cv2.imwrite('recorte.png', cv2.cvtColor(self.crop, cv2.COLOR_RGB2BGR))
                print("Recorte guardado como 'recorte.png'")

    def run(self):
        plt.tight_layout()
        
        while True:
            action = None
            done = False
            for direction, action_num in self.action_mapping.items():
                if keyboard.is_pressed(direction):
                    action = action_num
                    break
            
            if action is not None:
                self.obs, _, done, _, _ = self.env.step(action)
            #else:
                #self.obs, _, done, _, _ = self.env.step(0)  # Acción NOOP
            
            self.img1.set_data(self.obs)
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
            
            if done:
                self.obs, _ = self.env.reset()
            
            if keyboard.is_pressed('esc'):
                break

        plt.close()
        self.env.close()

if __name__ == "__main__":
    cortador = Cortador()
    cortador.run()