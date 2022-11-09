"""
# @Author: JuQi
# @Time  : 2022/11/9 16:10
# @E-mail: 18672750887@163.com
"""

import copy
import csv

import numpy as np


class Agent(object):
    def __init__(
            self,
            action_num: int,
            is_sample_action=False,
            is_BR=True,
            eta=1,
            is_using_history_regret=True,
            is_regret_plus=True,
            policy_update_mode='Fictitious'
    ):
        """
        __init__
        :param action_num: 策略的长度
        :param is_sample_action: 在交互的过程中，采取一个采样动作还是直接把策略返回
        :param is_BR: 是采用BR还是RM
        :param eta: RM的eta参数
        :param is_using_history_regret: 生成遗憾匹配策略时 采用历史的遗憾还是当前遗憾
        :param is_regret_plus: 历史遗憾是否只保留正值
        :param policy_update_mode: 可以取(Naive, Fictitious)
        Naive      得到的策略直接用于交互
        Fictitious 历史平均策略交互
        """
        self.action_num = action_num
        self.policy = np.random.random(self.action_num)
        self.policy = self.policy / np.sum(self.policy)
        self.history_policy = copy.deepcopy(self.policy)
        self.train_times = 1

        self.history_regret = np.zeros(self.action_num)
        self.update_policy = np.zeros(self.action_num)

        self.is_sample_action = is_sample_action
        self.is_BR = is_BR
        self.eta = eta
        self.is_using_history_regret = is_using_history_regret
        self.is_regret_plus = is_regret_plus
        self.policy_update_mode = policy_update_mode

    def get_history_policy(self) -> np.array:
        return self.history_policy / np.sum(self.history_policy)

    def get_interaction_policy(self) -> np.ndarray:
        if self.is_sample_action:
            tmp_action = np.random.choice(self.action_num, p=self.policy)
            tmp_policy = np.zeros(self.action_num)
            tmp_policy[tmp_action] = 1
            return tmp_policy
        else:
            return self.policy

    def get_update_policy(self, action_value: np.ndarray):
        if self.is_BR:
            self.update_policy = np.zeros(self.action_num)
            tmp_action_id = np.argmax(action_value)
            self.update_policy[tmp_action_id] = 1
        else:
            mean_value = np.sum(self.policy * action_value)
            now_regret = action_value - mean_value

            if self.is_using_history_regret:
                self.history_regret += now_regret
                cal_regret = self.history_regret
            else:
                cal_regret = now_regret

            if self.is_regret_plus:
                self.history_regret = np.maximum(self.history_regret, 0)
            else:
                pass

            tmp_cal_regret = np.maximum(cal_regret, 0)
            if np.sum(tmp_cal_regret) == 0:
                self.update_policy = np.ones(self.action_num) / self.action_num
            else:
                if self.eta == 0:
                    tmp_cal_regret[tmp_cal_regret > 0] = 1
                else:
                    tmp_cal_regret = np.power(tmp_cal_regret, self.eta)

                self.update_policy = tmp_cal_regret / np.sum(tmp_cal_regret)

    def policy_updates(self):
        if self.policy_update_mode == 'Naive':
            self.policy = self.update_policy
        elif self.policy_update_mode == 'Fictitious':
            self.train_times += 1
            alpha = 1 / self.train_times
            self.policy = (1 - alpha) * self.policy + alpha * self.update_policy

        self.history_policy += self.policy

    def get_setting(self):
        param_list = [
            self.is_sample_action.__str__(),
            self.is_BR.__str__(),
            self.eta.__str__(),
            self.is_using_history_regret.__str__(),
            self.is_regret_plus.__str__(),
            self.policy_update_mode.__str__()
        ]
        return param_list


class SymmetryGame(object):
    def __init__(
            self,
            game_mode,
            is_sample_action=False,
            is_BR=True,
            eta=1,
            is_using_history_regret=True,
            is_regret_plus=True,
            policy_update_mode='Fictitious'
    ):
        if game_mode == 'RPS':
            self.game_matrix = np.array(
                [
                    [0, 1, -1],
                    [-1, 0, 1],
                    [1, -1, 0]
                ]
            )
            self.action_num = 3
        else:
            self.game_matrix = np.random.random((game_mode, game_mode))
            for i in range(game_mode):
                self.game_matrix[i, i] = 0
                for j in range(i, game_mode):
                    self.game_matrix[i, j] = -self.game_matrix[j, i]
            self.action_num = game_mode

        self.player = Agent(
            self.action_num,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

    def iteration(self, iter_time):
        for _ in range(iter_time):
            interaction_policy = self.player.get_interaction_policy()
            action_value = np.matmul(self.game_matrix, interaction_policy.reshape((-1, 1)))
            action_value = action_value.reshape(-1)
            self.player.get_update_policy(action_value)
            self.player.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value = np.matmul(self.player.get_history_policy(), self.game_matrix)
            game_v = np.sum(action_value * self.player.get_history_policy())
        elif mode == 'now':
            action_value = np.matmul(self.player.policy, self.game_matrix)
            game_v = np.sum(action_value * self.player.policy)
        else:
            action_value = None
            game_v = 0
        epsilon = -np.min(action_value) - game_v
        return epsilon


if __name__ == '__main__':
    is_sample_action_list = [True, False]
    is_BR_list = [False]
    eta_list = [0, 0.5, 1, 2, 5]
    is_using_history_regret_list = [True, False]
    is_regret_plus_list = [True, False]
    policy_update_mode_list = ['Naive', 'Fictitious']

    # is_sample_action_list = [False]
    # is_BR_list = [False]
    # eta_list = [1]
    # is_using_history_regret_list = [False]
    # is_regret_plus_list = [False]
    # policy_update_mode_list = ['Naive']

    param_combination = [(x1, x2, x3, x4, x5, x6) for x1 in is_sample_action_list for x2 in is_BR_list for x3 in
                         eta_list for x4 in is_using_history_regret_list for x5 in is_regret_plus_list for x6 in
                         policy_update_mode_list]

    for act_len in [2, 4, 6, 8, 10]:
        print(act_len)
        for param in param_combination:
            print(param)
            recode_list = []
            his_epsilon_list = []
            now_epsilon_list = []
            for _ in range(20):
                rps = SymmetryGame(
                    game_mode=2,
                    is_sample_action=param[0],
                    is_BR=param[1],
                    eta=param[2],
                    is_using_history_regret=param[3],
                    is_regret_plus=param[4],
                    policy_update_mode=param[5]
                )
                if not recode_list:
                    recode_list.extend(rps.player.get_setting())

                rps.iteration(1000)
                his_epsilon_list.append(rps.get_epsilon('his'))
                now_epsilon_list.append(rps.get_epsilon('now'))

            recode_list.append(np.mean(now_epsilon_list))
            recode_list.append(np.mean(his_epsilon_list))

            with open('log/'+str(act_len)+'x'+str(act_len)+'_game.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(recode_list)
