"""
# @Author: JuQi
# @Time  : 2022/11/9 16:47
# @E-mail: 18672750887@163.com
"""
import csv
import numpy as np
from Games import FullCooperationGame, ZeroSumGame, SymmetryGame, NormalFromGame,RPSGame

if __name__ == '__main__':

    is_sample_action_list = [False]
    is_BR_list = [False]
    eta_list = [1, 16]
    is_using_history_regret_list = [True, False]
    is_regret_plus_list = [False]
    policy_update_mode_list = ['Naive', 'Fictitious']
    param_combination = [
        (x1, x2, x3, x4, x5, x6) for x1 in is_sample_action_list for x2 in is_BR_list for x3 in eta_list for x4 in
        is_using_history_regret_list for x5 in is_regret_plus_list for x6 in policy_update_mode_list
    ]
    itr_num = 10
    for act_len in [3]:
        print(act_len)
        for param in param_combination:
            print(param)
            game = RPSGame(
                action_len=act_len,
                is_sample_action=param[0],
                is_BR=param[1],
                eta=param[2],
                is_using_history_regret=param[3],
                is_regret_plus=param[4],
                policy_update_mode=param[5]
            )
            game.iteration(100)
