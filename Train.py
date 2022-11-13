"""
# @Author: JuQi
# @Time  : 2022/11/9 16:47
# @E-mail: 18672750887@163.com
"""
import csv
import numpy as np
from Games import FullCooperationGame, ZeroSumGame, SymmetryGame, NormalFromGame, FastZeroSumGame

if __name__ == '__main__':
    # is_sample_action_list = [True]
    # is_BR_list = [False]
    # eta_list = [-1]
    # is_using_history_regret_list = [True]
    # is_regret_plus_list = [False]
    # policy_update_mode_list = ['Naive']

    is_sample_action_list = [False]
    is_BR_list = [False]
    eta_list = [1, 16]
    is_using_history_regret_list = [True, False]
    is_regret_plus_list = [False]
    policy_update_mode_list = ['Naive', 'Fictitious']
    # param_combination = [
    #     (x1, x2, x3, x4, x5, x6) for x1 in is_sample_action_list for x2 in is_BR_list for x3 in eta_list for x4 in
    #     is_using_history_regret_list for x5 in is_regret_plus_list for x6 in policy_update_mode_list
    # ]
    param_combination = [
        (False, False, 16, True, False, 'Naive'),
        (False, False, 16, False, False, 'Fictitious')
    ]
    itr_num = 100
    for act_len in [8]:
        print(act_len)
        for param in param_combination:
            print(param)
            recode_list = []
            his_epsilon_list = np.zeros(itr_num)
            now_epsilon_list = np.zeros(itr_num)
            # his_value_list = []
            # now_value_list = []
            for i_game in range(2000):
                if i_game % 100 == 0:
                    print(i_game)
                game = FastZeroSumGame(
                    action_len=act_len
                    # action_len=act_len,
                    # is_sample_action=param[0],
                    # is_BR=param[1],
                    # eta=param[2],
                    # is_using_history_regret=param[3],
                    # is_regret_plus=param[4],
                    # policy_update_mode=param[5]
                )
                # if not recode_list:
                #     recode_list.extend(game.player1.get_setting())

                tmp_epsilon = game.iteration(itr_num)
                his_epsilon_list += tmp_epsilon
                # now_epsilon_list.append(game.get_epsilon('now'))
                # his_value_list.append(game.get_value('his'))
                # now_value_list.append(game.get_value('now'))

            recode_list.extend(list(his_epsilon_list))
            # recode_list.append(np.mean(his_epsilon_list))
            # recode_list.append(np.mean(now_value_list))
            # recode_list.append(np.mean(his_value_list))
            log_path = ''.join(['log/', game.name, '/', str(act_len), 'x', str(act_len), '_game', str(itr_num), '.csv'])
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(recode_list)
