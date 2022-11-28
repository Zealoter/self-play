"""
# @Author: JuQi
# @Time  : 2022/11/9 16:47
# @E-mail: 18672750887@163.com
"""
import numpy as np
from SP import Agent
import time
import os
import copy
import csv


class Game(object):
    def __init__(
            self,
            action_len
    ):
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/logCFR/', self.name, '_', now_time_str])

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


class NormalFromGame(Game):
    def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
                 is_regret_plus=True, policy_update_mode='Fictitious'):
        super().__init__(action_len)

        self.name = 'normal_form'
        for _ in range(2):
            self.game_matrices.append(np.random.randn(action_len, action_len))
            self.players.append(
                Agent(
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
        for _ in range(iter_time):
            interaction_policy1 = self.players[0].get_interaction_policy()
            interaction_policy2 = self.players[1].get_interaction_policy()

            action_value1 = np.matmul(self.game_matrices[0], interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrices[1])

            self.players[0].get_update_policy(action_value1)
            self.players[1].get_update_policy(action_value2)

            self.players[0].policy_updates()
            self.players[1].policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.players[0].get_history_policy(), self.game_matrices[1])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].get_history_policy().reshape((-1, 1)))
            game_v1 = np.sum(action_value1.reshape(-1) * self.players[0].get_history_policy())
            game_v2 = np.sum(action_value2 * self.players[1].get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.players[0].policy, self.game_matrices[1])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].policy.reshape((-1, 1)))
            game_v1 = np.sum(action_value1.reshape(-1) * self.players[0].policy)
            game_v2 = np.sum(action_value2 * self.players[1].policy)
        else:
            action_value1 = None
            action_value2 = None
            game_v1 = 0
            game_v2 = 0
        epsilon = np.max(action_value1) + np.max(action_value2) - game_v1 - game_v2
        return epsilon


class ZeroSumGame(Game):
    def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
                 is_regret_plus=True, policy_update_mode='Fictitious'):
        super().__init__(action_len)
        self.name = 'zero_sum'

        self.game_matrices.append(np.random.randn(action_len, action_len))

        for _ in range(2):
            self.players.append(
                Agent(
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


class FullCooperationGame(Game):
    def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
                 is_regret_plus=True, policy_update_mode='Fictitious'):
        super().__init__(action_len)
        self.name = 'full_cooperation'
        self.game_matrices.append(np.random.randn(action_len, action_len))

        for _ in range(2):
            self.players.append(
                Agent(
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
        for _ in range(iter_time):
            interaction_policy1 = self.players[0].get_interaction_policy()
            interaction_policy2 = self.players[1].get_interaction_policy()

            action_value1 = np.matmul(self.game_matrices[0], interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrices[0])

            self.players[0].get_update_policy(action_value1)
            self.players[1].get_update_policy(action_value2)

            self.players[0].policy_updates()
            self.players[1].policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.players[0].get_history_policy(), self.game_matrices[0])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].get_history_policy().reshape((-1, 1)))
            game_v = np.sum(action_value2 * self.players[1].get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.players[0].policy, self.game_matrices[0])
            action_value1 = np.matmul(self.game_matrices[0], self.players[1].policy.reshape((-1, 1)))
            game_v = np.sum(action_value2 * self.players[1].policy)
        else:
            action_value1 = None
            action_value2 = None
            game_v = 0
        epsilon = np.max(action_value1) + np.max(action_value2) - 2 * game_v
        return epsilon

    def get_value(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.players[0].get_history_policy(), self.game_matrices[0])
            game_v = np.sum(action_value2 * self.players[1].get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.players[0].policy, self.game_matrices[0])
            game_v = np.sum(action_value2 * self.players[1].policy)
        else:
            game_v = 0
        return game_v


class SymmetryGame(Game):
    def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
                 is_regret_plus=True, policy_update_mode='Fictitious'):
        super().__init__(action_len)
        self.name = 'symmetry'

        self.game_matrices.append(np.random.random((action_len, action_len)))

        for i in range(action_len):
            self.game_matrices[0][i, i] = 0
            for j in range(i, action_len):
                self.game_matrices[0][i, j] = -self.game_matrices[0][j, i]

        self.action_len = action_len

        self.players.append(
            Agent(
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
        for _ in range(iter_time):
            interaction_policy = self.players[0].get_interaction_policy()
            action_value = np.matmul(self.game_matrices[0], interaction_policy.reshape((-1, 1)))
            action_value = action_value.reshape(-1)
            self.players[0].get_update_policy(action_value)
            self.players[0].policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value = np.matmul(self.players[0].get_history_policy(), self.game_matrices[0])
            game_v = np.sum(action_value * self.players[0].get_history_policy())
        elif mode == 'now':
            action_value = np.matmul(self.players[0].policy, self.game_matrices[0])
            game_v = np.sum(action_value * self.players[0].policy)
        else:
            action_value = None
            game_v = 0
        epsilon = -np.min(action_value) - game_v
        return epsilon


class RPSGame(Game):
    def __init__(self, action_len, is_sample_action=False, is_BR=True, eta=1, is_using_history_regret=True,
                 is_regret_plus=True, policy_update_mode='Fictitious'):
        super().__init__(3)
        self.name = 'RPS'

        self.game_matrices.append(
            np.array(
                [
                    [0, 1, -1],
                    [-1, 0, 1],
                    [1, -1, 0]
                ]
            )
        )

        self.players.append(
            Agent(
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
        for _ in range(iter_time):
            interaction_policy = self.players[0].get_interaction_policy()
            action_value = np.matmul(self.game_matrices[0], interaction_policy.reshape((-1, 1)))
            action_value = action_value.reshape(-1)
            self.players[0].get_update_policy(action_value)
            self.players[0].policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value = np.matmul(self.players[0].get_history_policy(), self.game_matrices[0])
            game_v = np.sum(action_value * self.players[0].get_history_policy())
        elif mode == 'now':
            action_value = np.matmul(self.players[0].policy, self.game_matrices[0])
            game_v = np.sum(action_value * self.players[0].policy)
        else:
            action_value = None
            game_v = 0
        epsilon = -np.min(action_value) - game_v
        return epsilon
