import numpy as np
import matplotlib.pyplot as plt

#初始绘图
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)

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

line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

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
def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape  
    pi = np.zeros((m, n))
    exp_theta = np.exp(beta * theta)  
    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)  
    return pi

#初始策略
pi_0 = softmax_convert_into_pi_from_theta(theta_0)
# print(pi_0)


#随机获取下一个状态
def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == "up":
        action=0
        s_next = s - 3  
    elif next_direction == "right":
        action=1
        s_next = s + 1  
    elif next_direction == "down":
        action=2
        s_next = s + 3  
    elif next_direction == "left":
        action=3
        s_next = s - 1  
    return [action,s_next]

#移动agent到目标
def goal_maze_return_s_a(pi):
    s = 0  
    s_a_history = [[0,np.nan]]  
    while (1):  
        [action,next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1]=action
        s_a_history.append([next_s,np.nan])  
        if next_s == 8:  
            break
        else:
            s = next_s
    return s_a_history

#根据策略梯度法更新策略函数
def update_theta(theta, pi, s_a_history):
    eta = 0.1 
    T = len(s_a_history) - 1  
    [m, n] = theta.shape  
    delta_theta = theta.copy() 
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):  
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                N_i = len(SA_i)  
                N_ij = len(SA_ij)  
                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T
    new_theta = theta + eta * delta_theta
    return new_theta


#策略梯度法主函数
stop_epsilon = 10**-4 
theta = theta_0
pi = pi_0
is_continue = True
count = 1
while is_continue:  
    s_a_history = goal_maze_return_s_a(pi)  #随机运动
    new_theta = update_theta(theta, pi, s_a_history) 
    new_pi = softmax_convert_into_pi_from_theta(new_theta) 
    print(np.sum(np.abs(new_pi - pi)))  
    print("迷宫问题的步数为：" + str(len(s_a_history) - 1))
    count+=1
    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi

print(f"总迭代次数为：{count}")

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