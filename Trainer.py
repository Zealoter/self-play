"""
# @Author: JuQi
# @Time  : 2022/11/29 16:00
# @E-mail: 18672750887@163.com
"""
import os
import time
import Games


class Trainer(object):
    def __init__(self):
        self.name = 'zero_sum'
        self.now_path_str = os.getcwd()
        # 北京时间 东 8 区 +8
        self.now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time() + 8 * 60 * 60))
        self.result_file_path = ''.join([self.now_path_str, '/logCFR/', self.name, '_', self.now_time_str])

        if self.name == 'zero_sum':
            self.game = Games.ZeroSumGame()

