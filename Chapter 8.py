import numpy as np
import matplotlib.pyplot as plt


class Exe8_4():

    def __init__(self, start_state, goal, width, long, alpha, gamma, epsilon, n, maxi_iter):

        self.init_s = start_state  # should be a tuple.
        self.Miter = maxi_iter
        self.goal = goal
        self.width = width
        self.long = long
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = []
        self.n = n

        # a = (0, 1, 2, 3) * (width * long)
        s1 = range(long) * width
        s2 = []
        for i in range(width):
            s2 += [i] * long

        s = zip(s1, s2)
        sa = []

        for i in range(width*long):

            j = 0
            while j <= 3:
                tmp_sa = s[i]+(j, )
                sa.append(tmp_sa)
                j += 1

        self.Q = [[sa[i], 0.] for i in range(width * long * 4)]

    # def gridworld(self): it may be useful when we add block.
    #
    #
    #
    #     return

    def Dyna_Q(self):

        j = 0

        s = self.init_s

        while j <= self.Miter:

            # Q = [[(x1,y1,a1),value1], [(x2,y2,a2), value2], ...]


            A = self.epsilon_greed(s, self.Q, j, self.epsilon)

            tmp_R, tmp_next_s = self.conduct_A(s, A)

            s_a = s + (A,)

            index_s_a = [x for x, y in enumerate(self.Q) if y[0] == s_a][0]

            self.Q[index_s_a][-1] = self.Q[index_s_a][-1] + \
                                    self.alpha * \
                                    (tmp_R +
                                     self.gamma * self.max_a(self.Q, tmp_next_s) -
                                     self.Q[index_s_a][-1])

            self.model_Dyna(s, A, tmp_R, tmp_next_s)

            if tmp_next_s == self.goal:

                s = self.init_s

            else: s = tmp_next_s

            # print s

            for i in range(self.n):

                # print self.model
                # print len(self.model)
                # print np.random.randint(0, len(self.model), 1)
                tmp_s_A = self.model[np.random.randint(0, len(self.model), 1)[0]]

                tmp_s = tmp_s_A[:2]
                tmp_A = tmp_s_A[2]

                R_S = self.model_Dyna(tmp_s, tmp_A, 0, [])
                tmp_R = R_S[0]
                tmp_next_s = R_S[1:]

                index_s_a = [x for x, y in enumerate(self.Q) if y[0] == tmp_s_A[:3]][0]

                self.Q[index_s_a][-1] = self.Q[index_s_a][-1] + \
                                        self.alpha * \
                                            (tmp_R +
                                             self.gamma * self.max_a(self.Q, tmp_next_s) -
                                             self.Q[index_s_a][-1])

            j += 1

        # return self.Q

    def model_Dyna(self, s, A, tmp_R, tmp_next_s):

        if len(tmp_next_s) == 0:

            s_a = s + (A,)
            R_S = [x[3:] for x in self.model if x[:3] == s_a]

            return R_S[0]

        else:
            s_a_R_S = s + (A,) + (tmp_R,) + tmp_next_s

            if s_a_R_S not in self.model:

                self.model.append(s_a_R_S)

    def epsilon_greed(self, s, Q, j, epsilon):

        # Q = [[(x1,y1,a1),value1], [(x2,y2,a2), value2], ...]

        if j == 0:
            A = np.random.randint(0, 4, size=1)  # 0: up, 1: down, 2: left, 3:right
            A = int(A)
        else:

            action_value_s = [[x, y] for x, y in Q if x[:2] == s]

            tmp_value = [y for x, y in action_value_s]
            max_value = max(tmp_value)

            if max_value == 0:

                A = np.random.randint(0, 4, size=1)
                A = int(A)

            else:
                A = [x for x, y in action_value_s if y == max_value]

                if np.random.rand() <= 1 - epsilon:
                    A = int(A[0][-1])
                else:
                    A = np.random.randint(0, 4, size=1)
                    A = int(A)

        return A

    def conduct_A(self, s, A):

        # 0: up, 1: down, 2: left, 3:right

        # R = 0

        if A == 0:

            if s[1] + 1 <= self.width-1:

                s = list(s)
                s[1] += 1

                s = tuple(s)

                if s == self.goal:

                    R = 1
                else:
                    R = 0

            else:

                R = -1

        elif A == 1:

            if s[1] - 1 >= 0:

                s = list(s)
                s[1] -= 1

                s = tuple(s)

                if s == self.goal:

                    R = 1
                else:
                    R = 0
            else:

                R = -1

        elif A == 2:

            if s[0] - 1 >= 0:

                s = list(s)
                s[0] -= 1

                s = tuple(s)

                if s == self.goal:

                    R = 1
                else:
                    R = 0
            else:

                R = -1
        else:

            if s[0] + 1 <= self.long-1:

                s = list(s)
                s[0] += 1

                s = tuple(s)

                if s == self.goal:

                    R = 1
                else:
                    R = 0
            else:

                R = -1

        return R, s

    def max_a(self, Q, s):

        # tmp_S = []
        # for i in range(4):
        #     _, tmp_s = self.conduct_A(s, i)
        #     tmp_S.append(tmp_s+(i, ))

        action_value = []

        for i in range(4):
            # print tmp_S[i]
            # print len([y for x, y in Q if x[:2] == tmp_S[i]])
            # print [y for x, y in Q if x[:2] == tmp_S[i]]

            action_value.append([y for x, y in Q if x[:3] == (s+(i, ))][0])

        return max(action_value)

        # return [x for x, y in enumerate(action_value) if y == max(action_value)][0]

# (start_state, goal, width, long, alpha, gamma, epsilon, n, maxi_iter)
maze = Exe8_4((1, 1), (2, 3), 5, 5, 0.1, 1., 10 ** -2, 50, 5000)

maze.Dyna_Q()


for i in range(len(maze.Q)):

    print maze.Q[i]
# print maze.model

