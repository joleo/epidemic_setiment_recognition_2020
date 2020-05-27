# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/25 21:33
@Auth ： joleo
@File ：logs.py.py
"""
import os
import time
import logging
from config import *
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setlabel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
fh.setlabel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setlabel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(labelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)