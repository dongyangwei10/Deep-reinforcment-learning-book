import numpy as np
import matplotlib.pyplot as plt

#初始绘图
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)
# plt.plot([2.1, 3.3], [1.6, 1.9], color='red', linewidth=2)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)# draw the green cicle agent

# plt.show()

#初始参数，参数theta觉得策略pi
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                    [np.nan, 1, np.nan, 1],  # s1
                    [np.nan, np.nan, 1, 1],  # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1],  # s4
                    [1, np.nan, np.nan, np.nan],  # s5
                    [1, np.nan, np.nan, np.nan],  # s6
                    [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                    ])

#根据参数求取策略的函数
def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape  
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  
    pi = np.nan_to_num(pi)  
    return pi

#初始策略
pi_0 = simple_convert_into_pi_from_theta(theta_0)
[a, b] = theta_0.shape  
Q = np.random.rand(a, b) * theta_0 #*0.1

# ε-greedy法
def get_action(s, Q, epsilon, pi_0):#Q,pi:8*4;
    direction = ["up", "right", "down", "left"]
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])#从可选的action中，随机选择一个
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]#挑选最大价值的状态
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3
    return action

def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]  
    if next_direction == "up":
        s_next = s - 3  
    elif next_direction == "right":
        s_next = s + 1  
    elif next_direction == "down":
        s_next = s + 3  
    elif next_direction == "left":
        s_next = s - 1  
    return s_next

#Q learning价值更新
def QLearning(s, a, r, s_next,  Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q    

#一趟求解过程，求解过程中每一步都更新价值函数
def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0  
    a = a_next = get_action(s, Q, epsilon, pi)  
    s_a_history = [[0, np.nan]]  
    while (1):  
        a = a_next 
        s_a_history[-1][1] = a
        s_next = get_s_next(s, a, Q, epsilon, pi)
        s_a_history.append([s_next, np.nan])
        if s_next == 8:
            r = 1  
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
        Q = QLearning(s, a, r, s_next, Q, eta, gamma)
        if s_next == 8:  
            break
        else:
            s = s_next
    return [s_a_history, Q]    


#通过Q-Learning求解全过程
eta = 0.1  
gamma = 0.9  
epsilon = 0.5  
v = np.nanmax(Q, axis=1)  
is_continue = True
episode = 1
while is_continue: 
    print("当前回合:" + str(episode))
    epsilon = epsilon / 2
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
    new_v = np.nanmax(Q, axis=1) 
    print(np.sum(np.abs(new_v - v))) 
    v = new_v
    print("所需步数" + str(len(s_a_history) - 1) )
    episode = episode + 1
    if episode > 100:
        break    

#绘制动画
from matplotlib import animation
def init():
    line.set_data([], [])
    return (line,)
def animate(i):
    state = s_a_history[i][0] 
    x = (state % 3) + 0.5  
    y = 2.5 - int(state / 3) 
    line.set_data(x, y)
    return (line,)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(
    s_a_history), interval=200, repeat=False)

plt.show()    