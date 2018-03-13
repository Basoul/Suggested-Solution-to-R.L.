# # Exercise 4.5
#
# import numpy as np
# import math
#
#
# class exe4_5():
#     def __init__(self, num_cars, num_actions, gamma, reward, cost, lambda_):
#         # num_cars: number of cars in Park 1(2).
#         # num_actions: number of actions.
#         # gamma: discount rate.
#         self.n_c = num_cars
#         self.disount_rate = gamma
#         self.reward = reward
#
#         self.V = np.array([0.] * ((int(num_cars) + 1) ** 2))
#
#         self.A = range(-(int(num_actions) / 2), (int(num_actions) / 2) + 1)
#
#         # self.A = \
#         #     np.array(range(-5,6)*(num_cars ** 2)).reshape((num_cars ** 2, num_actions))
#         self.pi = np.array([0] * ((int(num_cars) + 1) ** 2))
#         self.cost = cost
#         self.lambda_ = lambda_
#
#         tmp1 = []
#         for i in range(int(num_cars) + 1):
#             tmp1.extend([i] * (int(num_cars) + 1))
#
#         tmp2 = range(int(num_cars) + 1) * (int(num_cars) + 1)
#
#         self.states = zip(tmp1, tmp2)
#
#         self.Poisson = lambda x, lambda_: float(lambda_ ** x) / math.factorial(x) * math.exp(-lambda_)
#
#     def E_exe4_5(self, V, current_pi, current_state):
#
#         car1, car2 = current_state
#         cost = self.cost * np.abs(current_pi)
#
#         tmp = 0
#         for next_state in self.states:
#
#             next_car1, next_car2 = next_state
#
#             tmp1 = 0
#
#             for rent_car in self.states:  # rent_car refer to the number of rented cars.
#
#                 rent_car1, rent_car2 = rent_car
#                 return_car1 = rent_car1 + next_car1 - car1 + current_pi
#                 return_car2 = rent_car2 + next_car2 - car2 - current_pi
#
#                 if (return_car1 >= 0) and (return_car2 >= 0):
#
#                     Pr1 = self.Poisson(return_car1, self.lambda_[2]) \
#                           * self.Poisson(rent_car1, self.lambda_[0])
#                     Pr2 = self.Poisson(return_car2, self.lambda_[3]) \
#                           * self.Poisson(rent_car2, self.lambda_[1])
#
#                     Pr = Pr1 * Pr2
#                 else:
#                     Pr = 0
#
#                 tmp1 += Pr * ((rent_car1 + rent_car2) * self.reward
#                               - cost
#                               + self.disount_rate * V[self.states.index(next_state)])
#
#             tmp += tmp1
#
#         return tmp
#
#     def pol_eva(self, theta):
#         # theta controls when stops.
#
#         diff_ = 1
#
#         while diff_ > theta:
#
#             tmp_V = self.V.copy()
#             diff_ = 0
#             for i in range(len(self.states)):
#                 v = tmp_V[i]
#
#                 self.V[i] = self.E_exe4_5(tmp_V, self.pi[i], self.states[i])
#                 diff_ = max(diff_, np.abs(v - self.V[i]))
#
#     def pol_imp(self):
#
#         pol_stable = True
#
#         for i in range(len(self.pi)):
#             old_action = self.pi[i]
#
#             tmp = [self.E_exe4_5(self.V,
#                                  j, self.states[i]) for j in self.A]
#
#             self.pi[i] = self.A[tmp.index(max(tmp))]
#
#             if old_action != self.pi[i]:
#                 pol_stable = False
#
#         print self.pi
#         return pol_stable
#
#
# car_problems = exe4_5(10., 5., 0.9, 10., 2., [3., 4., 3., 2.])
#
# pol_stable = False
#
# while pol_stable is not True:
#     car_problems.pol_eva(10 ** -2)
#     pol_stable = car_problems.pol_imp()
#
#     print pol_stable
#
# print car_problems.pi
# print car_problems.V
#
#
#












###################################################################################
#
# Exercise 4.9

import numpy as np
import matplotlib.pyplot as plt


class exe4_9():
    def __init__(self, ph, num_s, discount_rate, reward):
        self.goal = num_s
        self.s = np.arange(num_s + 1)  # 0 to num_s
        self.reward = reward
        self.ph = ph
        self.V = np.zeros(num_s + 1)
        # self.V[-1] = 1
        self.discount_rate = discount_rate

    def E_exe4_9(self, discount_rate, V, j, s):

        tmp_V = V
        s_ = [s + j, s - j]
        v = 0

        for num_s_ in range(len(s_)):

            if s_[num_s_] == self.goal:

                v += self.ph * (self.reward[1] + discount_rate * tmp_V[s_[num_s_]])
            else:

                if num_s_ == 0:

                    v += self.ph * (self.reward[0] + discount_rate * tmp_V[s_[num_s_]])
                else:

                    v += (1 - self.ph) * (self.reward[0] + discount_rate * tmp_V[s_[num_s_]])
        return v

    def val_ite(self):
        diff_ = 0

        for s in self.s:

            if (s != 0) and (s != 100):
                tmp_v = self.V[s]

                tmp_a_space = np.arange(min(s + 1, len(self.s) - s))  # 0 to min(s,100-s)

                tmp = [self.E_exe4_9(self.discount_rate,
                                     self.V,
                                     j, s) for j in tmp_a_space]

                # print s
                # print tmp

                self.V[s] = max(tmp)  # tmp.index(max(tmp))

                diff_ = max(np.abs(tmp_v - self.V[s]), diff_)

        return diff_  # , self.V


gamble = exe4_9(0.15, 100, 1, [0, 1])

# stopping_value = 10 ^ -3

# diff_ = 1

V_ = []

for i in range(33):
    gamble.val_ite()
    V_.append(gamble.V.copy())

for i in [0, 1, 2, 32]:

    print i
    plt.plot(range(1, 100), V_[i][1:-1])

plt.xlabel('Capital')
plt.ylabel('Value estimate')
plt.show()
