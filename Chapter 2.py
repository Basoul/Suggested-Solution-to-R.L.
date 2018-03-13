# Exercise 2.5


# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def sample_average_(old_Qt, R, step_size):
#     new_Qt = old_Qt + step_size * (R - old_Qt)
#
#     return new_Qt
#
#
# def select_actions(Qt, j, epsilon):
#     if j == 0:
#         A = np.random.randint(0, 10, size=1)
#         A = int(A)
#     else:
#
#         A = [x for x, y in enumerate(Qt) if y == max(Qt)]
#         u = np.random.rand()
#
#         if u <= 1 - epsilon:
#             A = int(A[0])
#         else:
#             A = np.random.randint(0, 10, size=1)
#             A = int(A)
#
#     return A
#
#
# iter_ = 200
# num_steps = 10000
#
# epsilon = 0.1
#
# # np.random.randn(10,1)
#
# all_results_ave = []
# all_results_alpha = []
#
# for i in range(iter_):
#
#
#     # stationary case
#     # q_star = np.array([np.random.normal(x, 1) for x in [0] * 10])
#
#     # nonstationary case
#     q_star = np.array([0] * 10)
#
#     Qt_ave = [0] * 10
#     Qt_alpha = [0] * 10
#     Nt = [0] * 10
#
#     tmp_result_ave = []
#     tmp_result_alpha = []
#
#     for j in range(num_steps):
#
#         tmp_norm = [np.random.normal(x, 1) for x in q_star]  # return at time t.
#
#         # the following variable q_star should be hided if one considers stationary case.
#         q_star = q_star+np.array([np.random.normal(x, 0.01) for x in [0] * 10])
#
#         A_ave = select_actions(Qt_ave, j, epsilon)
#         A_alpha = select_actions(Qt_alpha, j, epsilon)
#
#         # compute action value function
#         # average case
#         Nt[A_ave] += 1
#         R = tmp_norm[A_ave]
#         Qt_ave[A_ave] = sample_average_(Qt_ave[A_ave], R, 1.0 / Nt[A_ave])
#         tmp_result_ave.append(R)
#
#         # alpha=0.1 case
#         R = tmp_norm[A_alpha]
#         Qt_alpha[A_alpha] = sample_average_(Qt_alpha[A_alpha], R, 0.1)
#         tmp_result_alpha.append(R)
#
#     all_results_ave.append(tmp_result_ave)
#     all_results_alpha.append(tmp_result_alpha)
#
#
# all_results_1 = np.array(all_results_ave)
# all_results_2 = np.array(all_results_alpha)
#
# all_results_1 = np.mean(all_results_1, axis=0)
# all_results_2 = np.mean(all_results_2, axis=0)
#
# plt.plot(all_results_1)
# plt.plot(all_results_2, color='r')
# plt.show()
#
#


# Exercise 2.9


import numpy as np
import matplotlib.pyplot as plt


def sample_average_(old_Qt, R, step_size):
    new_Qt = old_Qt + step_size * (R - old_Qt)

    return new_Qt


def select_actions(Qt, j, epsilon):
    if j == 0:
        A = np.random.randint(0, 10, size=1)
        A = int(A)
    else:

        A = [x for x, y in enumerate(Qt) if y == max(Qt)]
        u = np.random.rand()

        if u <= 1 - epsilon:
            A = int(A[0])
        else:
            A = np.random.randint(0, 10, size=1)
            A = int(A)

    return A


def select_actionsUCB(Qt, j, Nt, c):
    if j == 0:
        A = np.random.randint(0, 10, size=1)
        A = int(A)
    else:
        extra_part = np.zeros(len(Nt))

        for i in range(len(Nt)):
            if Nt[i] != 0:
                extra_part[i] = c * np.sqrt(np.log(j) / Nt[i])

        tmp_Qt = Qt + extra_part
        A = [x for x, y in enumerate(tmp_Qt) if y == max(tmp_Qt)]

        A = int(A[0])

    return A


def soft_max(H):
    denominator_soft = sum(np.exp(H))

    numerator_soft = np.array([x / denominator_soft for x in np.exp(H)])

    return numerator_soft


iter_ = 1
num_steps = 2000  # 200000

step_size = 20

alpha = 0.1

epsilon_range = np.linspace(1. / 128, 1. / 4, step_size)  # for epsilon-greedy
alpha_range = np.linspace(1. / 32, 4, step_size)  # for gradient
c_range = np.linspace(1. / 20, 4, step_size)  # for UCB
Q0_range = np.linspace(1. / 4, 4, step_size)  # for greedy with optimistic

for i in range(iter_):

    q_star = np.array([np.random.normal(x, 1) for x in [0] * 10])
    # nonstationary case
    # q_star = np.zeros(10)

    Qt_UCB = np.zeros((step_size, 10))
    Qt_epsilon = np.zeros((step_size, 10))
    Qt_G = np.zeros(step_size)

    Qt_greedy = list(Q0_range)*10
    Qt_greedy = np.array(Qt_greedy).reshape((10, step_size)).T

    Nt = np.zeros((step_size, 10))
    Nt_UCB = np.zeros((step_size, 10))

    H = np.zeros((step_size, 10))

    p = np.zeros((step_size, 10)) + 1./10

    tmp_result_UCB = np.zeros((num_steps, step_size))
    tmp_result_epsilon = np.zeros((num_steps, step_size))
    tmp_result_G = np.zeros((num_steps, step_size))
    tmp_result_greedy = np.zeros((num_steps, step_size))

    for j in range(num_steps):
        tmp_norm = [np.random.normal(x, 1) for x in q_star]  # return at time t.

        # q_star = q_star + np.array([np.random.normal(x, 0.01) for x in [0] * 10])  # nonstationary case.

        for i in range(step_size):

            # UCB
            A_UCB = select_actionsUCB(Qt_UCB[i], j, Nt_UCB[i], c_range[i])
            Nt_UCB[i][A_UCB] += 1
            R = tmp_norm[A_UCB]
            Qt_UCB[i][A_UCB] = sample_average_(Qt_UCB[i][A_UCB], R, 1.0 / Nt_UCB[i][A_UCB])
            tmp_result_UCB[j][i] = R

            # epsilon-greedy
            A_epsilon = select_actions(Qt_epsilon[i], j, epsilon_range[i])
            R = tmp_norm[A_epsilon]
            Qt_epsilon[i][A_epsilon] = sample_average_(Qt_epsilon[i][A_epsilon], R, 0.1)
            tmp_result_epsilon[j][i] = R

            # Gradient method
            A_G = int(np.random.choice(np.arange(0, 10), p = p[i]))  # incomplete: without selection weights.
            R = tmp_norm[A_G]
            Qt_G[i] = sample_average_(Qt_G[i], R, 1. / (j + 1))
            for k in range(len(p)):

                if k == A_G:
                    H[i][A_G] = H[i][A_G] + alpha_range[i] * (R - Qt_G[i]) * (1 - p[i][A_G])
                else:
                    H[i][A_G] = H[i][A_G] - alpha_range[i] * (R - Qt_G[i]) * p[i][A_G]

            p[i] = soft_max(H[i])
            tmp_result_G[j][i] = R

            # Optimistic method.
            A_greedy = select_actions(Qt_greedy[i], j, 0)
            R = tmp_norm[A_greedy]
            Qt_greedy[i][A_greedy] = sample_average_(Qt_greedy[i][A_greedy], R, 0.1)
            tmp_result_greedy[j][i] = R

print Qt_UCB
print Qt_epsilon
print Qt_G
print Qt_greedy

all_results_UCB = np.mean(tmp_result_UCB[-1000:, :], axis=0)
all_results_epsilon = np.mean(tmp_result_epsilon[-1000:, :], axis=0)
all_results_G = np.mean(tmp_result_G[-1000:, :], axis=0)
all_results_greedy = np.mean(tmp_result_greedy[-1000:, :], axis=0)

plt.plot(np.log(c_range), all_results_UCB, color='b')

plt.plot(np.log(epsilon_range), all_results_epsilon, color='r')
plt.plot(np.log(alpha_range), all_results_G, color='g')
plt.plot(np.log(Q0_range), all_results_greedy, color='k')

plt.show()

'''
for k in range(2):
    all_results = []
    for i in range(iter_):
        Qt = [0] * 10
        Nt = [0] * 10
        tmp_norm = [np.random.normal(x,1,num_steps) for x in q_star]
        tmp_result = []
        for j in range(num_steps):
            if j == 0:
                A = np.random.randint(0,10,size=1)
                A = int(A)
            else:

                A = [x for x,y in enumerate(Qt) if y == max(Qt)]
                A = int(A[0])

            if k == 0:
                Nt[A] += 1
                R = tmp_norm[A][j]
                Qt[A] = sample_average_(Qt[A], R, 1.0/Nt[A])
            else:
                R = tmp_norm[A][j]
                Qt[A] = sample_average_(Qt[A], R, 0.1)

            tmp_result.append(R)
        all_results.append(tmp_result)
    all_results_.append(all_results)

all_results_1 = np.array(all_results_[0])
all_results_2 = np.array(all_results_[1])

all_results_1 = np.mean(all_results_1, axis=0)
all_results_2 = np.mean(all_results_2, axis=0)

plt.plot(all_results_1)
plt.plot(all_results_2,color='r')
plt.show()

'''
