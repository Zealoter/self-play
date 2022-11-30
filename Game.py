"""
# @Author: JuQi
# @Time  : 2022/11/29 16:32
# @E-mail: 18672750887@163.com
"""
import numpy as np
from Agent import SPAgent
import time
import os
import copy
import csv


class Game(object):
    def __init__(
            self,
            action_len
    ):
        self.action_len = action_len

        self.game_matrices = []
        self.players = []

    def iteration(self, iter_time):
        pass

    def get_epsilon(self, mode='his'):
        pass

    def reset(self):
        for tmp_player in self.players:
            tmp_player.reset()
        for tmp_i in range(len(self.game_matrices)):
            self.game_matrices[tmp_i] = np.random.randn(self.action_len, self.action_len)


class ZeroSumGame(Game):
    # def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
    #              is_regret_plus=True, policy_update_mode='Fictitious'):
    def __init__(self, config):
        super().__init__(action_len)
        self.name = 'zero_sum'

        self.game_matrices.append(np.random.randn(action_len, action_len))

        for _ in range(2):
            self.players.append(
                SPAgent(
                    self.action_len,
                    is_sample_action,
                    is_BR,
                    eta,
                    is_using_history_regret,
                    is_regret_plus,
                    policy_update_mode
                )
            )

    def iteration(self, iter_time):
        epsilon_list = np.zeros(iter_time)
        for i_itr in range(iter_time):
            interaction_policy1 = self.players[0].get_interaction_policy()
            interaction_policy2 = self.players[1].get_interaction_policy()

            action_value1 = np.matmul(self.game_matrices[0], interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrices[0])

            self.players[0].get_update_policy(action_value1)
            self.players[1].get_update_policy(-action_value2)

            self.players[0].policy_updates()
            self.players[1].policy_updates()

            epsilon_list[i_itr] = self.get_epsilon('his')
        return epsilon_list

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.players[0].get_history_policy(), self.game_matrices[0])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].get_history_policy().reshape((-1, 1)))

        elif mode == 'now':
            action_value2 = np.matmul(self.players[0].policy, self.game_matrices[0])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].policy.reshape((-1, 1)))

        else:
            action_value1 = None
            action_value2 = None

        epsilon = np.max(action_value1) - np.min(action_value2)
        return epsilon
