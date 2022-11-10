"""
# @Author: JuQi
# @Time  : 2022/11/9 16:47
# @E-mail: 18672750887@163.com
"""
import numpy as np
from SP import Agent


class ZeroSumGame(object):
    def __init__(
            self,
            action_len,
            is_sample_action=False,
            is_BR=True,
            eta=1,
            is_using_history_regret=True,
            is_regret_plus=True,
            policy_update_mode='Fictitious'
    ):
        self.name = 'zero_sum'
        self.game_matrix = np.random.randn(action_len, action_len)
        self.action_num = action_len
        self.player1 = Agent(
            self.action_num,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.player2 = Agent(
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
            interaction_policy1 = self.player1.get_interaction_policy()
            interaction_policy2 = self.player2.get_interaction_policy()

            action_value1 = np.matmul(self.game_matrix, interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrix)

            self.player1.get_update_policy(action_value1)
            self.player2.get_update_policy(-action_value2)

            self.player1.policy_updates()
            self.player2.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.get_history_policy().reshape((-1, 1)))

            # game_v = np.sum(action_value2 * self.player2.get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.player1.policy, self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.policy.reshape((-1, 1)))
            # game_v = np.sum(action_value2 * self.player2.policy)
        else:
            action_value1 = None
            action_value2 = None
            # game_v = 0
        epsilon = np.max(action_value1) - np.min(action_value2)
        return epsilon


class FullCooperationGame(object):
    def __init__(
            self,
            action_len,
            is_sample_action=False,
            is_BR=True,
            eta=1,
            is_using_history_regret=True,
            is_regret_plus=True,
            policy_update_mode='Fictitious'
    ):
        self.name = 'full_cooperation'
        self.game_matrix = np.random.randn(action_len, action_len)
        self.action_num = action_len
        self.player1 = Agent(
            self.action_num,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.player2 = Agent(
            self.action_num,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )
        self.name = 'zero_sum'

    def iteration(self, iter_time):
        for _ in range(iter_time):
            interaction_policy1 = self.player1.get_interaction_policy()
            interaction_policy2 = self.player2.get_interaction_policy()

            action_value1 = np.matmul(self.game_matrix, interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrix)

            self.player1.get_update_policy(action_value1)
            self.player2.get_update_policy(action_value2)

            self.player1.policy_updates()
            self.player2.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.get_history_policy().reshape((-1, 1)))
            game_v = np.sum(action_value2 * self.player2.get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.player1.policy, self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.policy.reshape((-1, 1)))
            game_v = np.sum(action_value2 * self.player2.policy)
        else:
            action_value1 = None
            action_value2 = None
            game_v = 0
        epsilon = np.max(action_value1) + np.min(action_value2) - 2 * game_v
        return epsilon


class SymmetryGame(object):
    def __init__(
            self,
            action_num,
            is_sample_action=False,
            is_BR=True,
            eta=1,
            is_using_history_regret=True,
            is_regret_plus=True,
            policy_update_mode='Fictitious'
    ):
        self.name = 'symmetry'
        if action_num == 'RPS':
            self.game_matrix = np.array(
                [
                    [0, 1, -1],
                    [-1, 0, 1],
                    [1, -1, 0]
                ]
            )
            self.action_num = 3
        else:
            self.game_matrix = np.random.random((action_num, action_num))
            for i in range(action_num):
                self.game_matrix[i, i] = 0
                for j in range(i, action_num):
                    self.game_matrix[i, j] = -self.game_matrix[j, i]
            self.action_num = action_num

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
