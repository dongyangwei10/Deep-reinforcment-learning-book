import numpy as np
import matplotlib.pyplot as plt
import gym

ENV = 'CartPole-v0' 
NUM_DIZITIZED = 6 

env = gym.make(ENV)  
observation = env.reset()  

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED))]
    return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

digitize_state(observation)
# print(digitize_state(observation))