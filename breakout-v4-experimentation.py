import gym
import numpy as np
from scipy.misc import imresize
from PIL import Image

env = gym.make('Breakout-v4')
obs = env.reset()

obs = obs.mean(axis=2).astype(np.uint8)
im = Image.fromarray(obs)
roi_box = (8, 93, 152, 194)
roi = im.crop(roi_box)
bbox = roi.getbbox()

# roi = roi.crop(bbox)
print(bbox)
roi.save("img1.png")

# print(obs)
