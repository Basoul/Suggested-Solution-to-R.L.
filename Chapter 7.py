# Exercise 7.2

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class exe7_2():
#     def __init__(self, S0, ph, num_s, discount_rate, reward, alpha, n):
#         self.goal = num_s
#         self.s = np.arange(num_s + 1)  # 0 to num_s
#         self.reward = reward
#         self.ph = ph
#         self.V = np.zeros(num_s + 1)
#         # self.V[-1] = 1
#         self.discount_rate = discount_rate
#         self.alpha = alpha
#         self.init_S = S0
#         self.n = n
#
#     def TD_n(self):
#         # n = 1: TD.
#
#         T = 10 ** 7
#
#         tmp_s = self.init_S
#         S_states = [tmp_s]
#         R_states = [0]
#
#         for t in range(10 ** 7):
#
#             if t < T:
#
#                 tmp_a_space = \
#                     np.arange(min(tmp_s + 1, len(self.s) - tmp_s))  # 0 to min(s,100-s)
#
#                 tmp_a = np.random.choice(tmp_a_space, 1)
#
#                 if np.random.rand() <= self.ph:
#                     tmp_R = tmp_a
#                 else:
#                     tmp_R = -tmp_a
#
#                 tmp_s += tmp_R
#
#                 if tmp_s == 100: R_states.append(self.reward[1])
#                 else: R_states.append(self.reward[0])
#
#                 S_states.append(int(tmp_s))
#
#                 if (tmp_s == 0) or (tmp_s == 100):
#                     T = t + 1
#
#             tao = t - self.n + 1
#
#             if tao >= 0:
#
#                 G = self.n_sum(tao + 1, min(tao + self.n, T), R_states)
#
#                 if tao + self.n < T:
#
#                     G = G + self.discount_rate ** self.n * self.V[S_states[tao+self.n]]
#
#                 self.V[S_states[tao]] += self.alpha * (G - self.V[S_states[tao]])
#
#                 if tao == T - 1:
#                     break
#
#     def n_sum(self, start, end, R_states):
#
#         # n = end-start+1
#
#         gammas = np.array([(self.discount_rate ** i) for i in range(end - (start - 1))])
#
#
#         part_R = np.array(R_states[start:end + 1])
#
#         return np.sum(gammas * part_R)
#
# gamble = exe7_2(50, 0.5, 100, 1, [0, 1], 0.1, 3)
#
# # stopping_value = 10 ^ -3
#
# # diff_ = 1
#
# V_ = []
#
# for i in range(2000):
#     gamble.TD_n()
#     V_.append(gamble.V.copy())
#
# plt.plot(range(1, 100), V_[-1][1:-1])
#
#
#
# gamble = exe7_2(50, 0.5, 100, 1, [0, 1], 0.1, 1)
#
# # stopping_value = 10 ^ -3
#
# # diff_ = 1
#
# V_ = []
#
# for i in range(2000):
#     gamble.TD_n()
#     V_.append(gamble.V.copy())
#
# plt.plot(range(1, 100), V_[-1][1:-1],'r')
#
# plt.xlabel('Capital')
# plt.ylabel('Value estimate')
# plt.show()
#

# Exercise 7.8

import numpy as np
import matplotlib.pyplot as plt


class exe7_8():
    def __init__(self, init_S, num_states, t_pi, b_pi, n, rewards, alpha, gamma):

        self.states = range(num_states)  # contains two terminal states (the most left and right)
        self.t_pi = t_pi
        self.b_pi = b_pi
        self.reward = rewards
        self.V = [0] * num_states
        self.init_S = init_S
        self.n = n
        self.alpha = alpha
        self.discount_rate = gamma

    def off_police_TD_n(self, new):
        # n = 1: TD.

        T = 10 ** 7

        tmp_s = self.init_S
        S_states = [tmp_s]
        A_states = [0]
        R_states = [0]

        for t in range(10 ** 7):

            if t < T:

                if np.random.rand() <= self.b_pi:
                    tmp_a = -1  # left
                else:
                    tmp_a = 1  # right

                A_states.append(tmp_a)

                tmp_s += tmp_a

                if tmp_s == self.states[-1]:
                    R_states.append(self.reward[1])
                else:
                    R_states.append(self.reward[0])

                S_states.append(int(tmp_s))

                if (tmp_s == 0) or (tmp_s == self.states[-1]):
                    T = t + 1

            tao = t - self.n + 1

            if tao >= 0:

                rho = self.n_times(tao + 1, min(tao + self.n - 1, T - 1), A_states)

                G = self.n_sum(tao + 1, min(tao + self.n, T), R_states)

                if tao + self.n < T:
                    G = G + self.discount_rate ** self.n * self.V[S_states[tao + self.n]]

                    if new == True:

                        rho_t = self.n_times(tao + 1, tao + 1, A_states)
                        G = rho_t * G + (1. - rho_t) * self.V[S_states[tao]]

                self.V[S_states[tao]] += self.alpha * rho * (G - self.V[S_states[tao]])

                if tao == T - 1:
                    break

    def n_times(self, start, end, A_states):

        tmp_numerator = A_states[start]
        tmp_denominator = A_states[start]

        for i in range(start + 1, end + 1):

            if A_states[i] == -1:

                tmp_numerator *= self.t_pi
                tmp_denominator *= self.b_pi
            else:

                tmp_denominator *= 1. - self.t_pi
                tmp_numerator *= 1. - self.b_pi

        return tmp_numerator / tmp_denominator

    def n_sum(self, start, end, R_states):

        # n = end-start+1

        gammas = np.array([(self.discount_rate ** i) for i in range(end - (start - 1))])

        part_R = np.array(R_states[start:end + 1])

        return np.sum(gammas * part_R)


# init_S, num_states, t_pi, b_pi, n, rewards, alpha, gamma
gamble = exe7_8(4, 7, 0, 0.5, 3, [0, 1], 0.1, 0.9)

# stopping_value = 10 ^ -3

# diff_ = 1

V_ = []

for i in range(200):
    gamble.off_police_TD_n(new=False)

    V_.append(np.array(gamble.V).copy())

plt.plot(range(1, 6), V_[-1][1:-1])


gamble_ = exe7_8(4, 7, 0, 0.5, 3, [0, 1], 0.1, 0.9)

for i in range(200):
    gamble_.off_police_TD_n(new=True)

    V_.append(np.array(gamble_.V).copy())

plt.plot(range(1, 6), V_[-1][1:-1],'r')

plt.xlabel('states')
plt.ylabel('Value estimate')
plt.show()
