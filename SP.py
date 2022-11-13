"""
# @Author: JuQi
# @Time  : 2022/11/9 16:10
# @E-mail: 18672750887@163.com
"""

import copy
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
        # self.policy = np.ones(self.action_num)
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
                elif self.eta < 0:
                    tmp_cal_regret[tmp_cal_regret > 0] = np.power(tmp_cal_regret[tmp_cal_regret > 0], self.eta)

                else:
                    tmp_cal_regret = np.power(tmp_cal_regret, self.eta)
                if self.eta < 1:
                    self.eta += 0.01
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


class FastAgent(object):
    def __init__(
            self,
            action_num: int,

    ):
        """
        __init__
        :param action_num: 策略的长度

        """
        self.action_num = action_num
        self.policy = np.random.random(self.action_num)
        # self.policy = np.ones(self.action_num)
        self.policy = self.policy / np.sum(self.policy)
        self.history_policy = copy.deepcopy(self.policy)
        self.train_times = 1

        self.history_regret = np.zeros(self.action_num)
        self.update_policy = np.zeros(self.action_num)

    def get_history_policy(self) -> np.array:
        return self.history_policy / np.sum(self.history_policy)

    def get_interaction_policy(self) -> np.ndarray:
        self.train_times += 1
        return self.policy

    def get_update_policy(self, action_value: np.ndarray):
        mean_value = np.sum(self.policy * action_value)
        now_regret = action_value - mean_value
        alpha = 1 - 2 / (self.train_times + 1)

        self.history_regret = (1 - alpha) * self.history_regret + alpha * now_regret

        tmp_cal_regret = np.maximum(self.history_regret, 0)
        if np.sum(tmp_cal_regret) == 0:
            self.update_policy = np.ones(self.action_num) / self.action_num
        else:
            self.update_policy = tmp_cal_regret / np.sum(tmp_cal_regret)

    def policy_updates(self):
        alpha = 2 / (self.train_times + 1)
        self.policy = (1 - alpha) * self.policy + alpha * self.update_policy

        self.history_policy += self.policy

