"""
# @Author: JuQi
# @Time  : 2022/11/30 17:22
# @E-mail: 18672750887@163.com
"""
DEFAULT_CONFIG = {
    'train_config':
        {
            "cpu_num"        : 1,
            "train_trail_num": 10,
            "train_itr_num"  : 1000,
            'action_num'     : 3
        },
    "Agent_config":
        {
            'is_sample_action'       : False,
            'is_BR'                  : True,
            'eta'                    : 1,
            'is_using_history_regret': True,
            'is_regret_plus'         : True,
            'policy_update_mode'     : 'Fictitious'
        },
    'Game_config' :
        {
            'game_payoff_matrix_mode': 'normal_distribution',
            'means'                  : 0,
            'variance'               : 1
        }
}
