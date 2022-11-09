"""
# @Author: JuQi
# @Time  : 2022/11/9 16:47
# @E-mail: 18672750887@163.com
"""
import csv
import numpy as np
from Games import ZeroSumGame

if __name__ == '__main__':
    is_sample_action_list = [True, False]
    is_BR_list = [False]
    eta_list = [0, 0.5, 1, 5, 100]
    is_using_history_regret_list = [True, False]
    is_regret_plus_list = [True, False]
    policy_update_mode_list = ['Naive', 'Fictitious']

    # is_sample_action_list = [False]
    # is_BR_list = [False]
    # eta_list = [1]
    # is_using_history_regret_list = [False]
    # is_regret_plus_list = [False]
    # policy_update_mode_list = ['Naive']

    param_combination = [
        (x1, x2, x3, x4, x5, x6) for x1 in is_sample_action_list for x2 in is_BR_list for x3 in eta_list for x4 in
        is_using_history_regret_list for x5 in is_regret_plus_list for x6 in policy_update_mode_list
    ]
    for act_len in [2, 4, 6, 8, 10]:
        print(act_len)
        for param in param_combination:
            print(param)
            recode_list = []
            his_epsilon_list = []
            now_epsilon_list = []
            for _ in range(20):
                rps = ZeroSumGame(
                    action_len=2,
                    is_sample_action=param[0],
                    is_BR=param[1],
                    eta=param[2],
                    is_using_history_regret=param[3],
                    is_regret_plus=param[4],
                    policy_update_mode=param[5]
                )
                if not recode_list:
                    recode_list.extend(rps.player1.get_setting())

                rps.iteration(1000)
                his_epsilon_list.append(rps.get_epsilon('his'))
                now_epsilon_list.append(rps.get_epsilon('now'))

            recode_list.append(np.mean(now_epsilon_list))
            recode_list.append(np.mean(his_epsilon_list))

            with open('log/zero_sum/' + str(act_len) + 'x' + str(act_len) + '_game.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(recode_list)
