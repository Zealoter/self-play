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


class NormalFromGame(object):
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
        self.name = 'normal_form'
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/logCFR/', self.name, '_', now_time_str])
        self.game_matrix1 = np.random.randn(action_len, action_len)
        self.game_matrix2 = np.random.randn(action_len, action_len)
        self.action_len = action_len
        self.player1 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.player2 = Agent(
            self.action_len,
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

            action_value1 = np.matmul(self.game_matrix1, interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrix2)

            self.player1.get_update_policy(action_value1)
            self.player2.get_update_policy(action_value2)

            self.player1.policy_updates()
            self.player2.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.player1.get_history_policy(), self.game_matrix2)
            action_value1 = np.matmul(self.game_matrix1, self.player2.get_history_policy().reshape((-1, 1)))
            game_v1 = np.sum(action_value1.reshape(-1) * self.player1.get_history_policy())
            game_v2 = np.sum(action_value2 * self.player2.get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.player1.policy, self.game_matrix2)
            action_value1 = np.matmul(self.game_matrix1, self.player2.policy.reshape((-1, 1)))
            game_v1 = np.sum(action_value1.reshape(-1) * self.player1.policy)
            game_v2 = np.sum(action_value2 * self.player2.policy)
        else:
            action_value1 = None
            action_value2 = None
            game_v1 = 0
            game_v2 = 0
        epsilon = np.max(action_value1) + np.max(action_value2) - game_v1 - game_v2
        return epsilon

    def reset(self):
        self.game_matrix1 = np.random.randn(self.action_len, self.action_len)
        self.game_matrix2 = np.random.randn(self.action_len, self.action_len)
        self.player1.reset()
        self.player2.reset()


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
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/logCFR/', self.name, '_', now_time_str])
        self.game_matrix = np.random.randn(action_len, action_len)
        self.action_len = action_len
        self.player1 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.player2 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

    def iteration(self, iter_time):
        epsilon_list = np.zeros(iter_time)
        for i_itr in range(iter_time):
            interaction_policy1 = self.player1.get_interaction_policy()
            interaction_policy2 = self.player2.get_interaction_policy()

            action_value1 = np.matmul(self.game_matrix, interaction_policy2.reshape((-1, 1)))
            action_value1 = action_value1.reshape(-1)

            action_value2 = np.matmul(interaction_policy1, self.game_matrix)

            self.player1.get_update_policy(action_value1)
            self.player2.get_update_policy(-action_value2)

            self.player1.policy_updates()
            self.player2.policy_updates()

            epsilon_list[i_itr] = self.get_epsilon('his')
        return epsilon_list

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.get_history_policy().reshape((-1, 1)))

        elif mode == 'now':
            action_value2 = np.matmul(self.player1.policy, self.game_matrix)
            action_value1 = np.matmul(self.game_matrix, self.player2.policy.reshape((-1, 1)))

        else:
            action_value1 = None
            action_value2 = None

        epsilon = np.max(action_value1) - np.min(action_value2)
        return epsilon

    def reset(self):
        self.game_matrix = np.random.randn(self.action_len, self.action_len)
        self.player1.reset()
        self.player2.reset()


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
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/logCFR/', self.name, '_', now_time_str])
        self.game_matrix = np.random.randn(action_len, action_len)
        self.action_len = action_len
        self.player1 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.player2 = Agent(
            self.action_len,
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
        epsilon = np.max(action_value1) + np.max(action_value2) - 2 * game_v
        return epsilon

    def get_value(self, mode='his'):
        if mode == 'his':
            action_value2 = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            game_v = np.sum(action_value2 * self.player2.get_history_policy())
        elif mode == 'now':
            action_value2 = np.matmul(self.player1.policy, self.game_matrix)
            game_v = np.sum(action_value2 * self.player2.policy)
        else:
            game_v = 0
        return game_v

    def reset(self):
        self.game_matrix = np.random.randn(self.action_len, self.action_len)
        self.player1.reset()
        self.player2.reset()


class SymmetryGame(object):
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
        self.name = 'symmetry'
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([now_path_str, '/logCFR/', self.name, '_', now_time_str])
        self.game_matrix = np.random.random((action_len, action_len))
        for i in range(action_len):
            self.game_matrix[i, i] = 0
            for j in range(i, action_len):
                self.game_matrix[i, j] = -self.game_matrix[j, i]
        self.action_len = action_len

        self.player1 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

    def iteration(self, iter_time):
        for _ in range(iter_time):
            interaction_policy = self.player1.get_interaction_policy()
            action_value = np.matmul(self.game_matrix, interaction_policy.reshape((-1, 1)))
            action_value = action_value.reshape(-1)
            self.player1.get_update_policy(action_value)
            self.player1.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            game_v = np.sum(action_value * self.player1.get_history_policy())
        elif mode == 'now':
            action_value = np.matmul(self.player1.policy, self.game_matrix)
            game_v = np.sum(action_value * self.player1.policy)
        else:
            action_value = None
            game_v = 0
        epsilon = -np.min(action_value) - game_v
        return epsilon

    def reset(self):
        self.game_matrix = np.random.randn(self.action_len, self.action_len)
        self.player1.reset()


class RPSGame(object):
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
        self.name = 'RPS'
        now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))

        self.game_matrix = np.array(
            [
                [0, 1, -1],
                [-1, 0, 1],
                [1, -1, 0]
            ]
        )
        self.action_len = 3

        self.player1 = Agent(
            self.action_len,
            is_sample_action,
            is_BR,
            eta,
            is_using_history_regret,
            is_regret_plus,
            policy_update_mode
        )

        self.result_file_path = ''.join(
            [now_path_str, '/log/', self.name, '/', self.player1.get_setting(), '/', now_time_str]
        )

    def iteration(self, iter_time):
        for _ in range(iter_time):
            interaction_policy = self.player1.get_interaction_policy()
            action_value = np.matmul(self.game_matrix, interaction_policy.reshape((-1, 1)))
            action_value = action_value.reshape(-1)
            self.player1.get_update_policy(action_value)
            self.player1.policy_updates()

    def get_epsilon(self, mode='his'):
        if mode == 'his':
            action_value = np.matmul(self.player1.get_history_policy(), self.game_matrix)
            game_v = np.sum(action_value * self.player1.get_history_policy())
        elif mode == 'now':
            action_value = np.matmul(self.player1.policy, self.game_matrix)
            game_v = np.sum(action_value * self.player1.policy)
        else:
            action_value = None
            game_v = 0
        epsilon = -np.min(action_value) - game_v
        return epsilon

    def reset(self):
        self.game_matrix = np.random.randn(self.action_len, self.action_len)
        self.player1.reset()

    def train(self, train_num: int, save_interval: int, log_interval: int):

        os.makedirs(self.result_file_path)
        log_path = ''.join(
            [self.result_file_path, '/', str(train_num), '_', str(log_interval), '.csv']
        )
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(recode_list)

            for itr in range(1, train_num + 1):

                self.reset()

                if itr % save_interval == 0:
                    CFR_cur = get_policy()
                    self.save_model(itr, CFR_cur)
                if itr % log_interval == 0:
                    CFR_cur = get_policy()
                    get_result_inline(itr, CFR_cur, self.result_file_path, self.prior_state, self.poker_game)
