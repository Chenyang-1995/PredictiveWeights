import math
import numpy as np
import numpy.random as random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt

from graph import *
from data_preprocess import *



def robustness_test(csv_fname_list,load_mode='least-degree',arriving_order='random'):

    print("Start {}!".format(csv_fname_list))
    df_list = []
    num_loop = 4
    PW_ratio_list = []
    G_ratio_list = []
    IPW_ratio_list = []
    l1_norm_list = []

    R_ratio_list = []
    num_days = len(csv_fname_list)
    initial_df = read_data(csv_fname_list[0])
    initial_graph = Graph(initial_df,load_mode=load_mode)

    initial_graph.train_ad_weights(train_ratio=1.0)

    for day_id in range(1,num_days):
        logging_str = ''
        logging_str += '----day_{}----\n'.format( day_id + 1)
        today_df = read_data(csv_fname_list[day_id])

        today_graph = Graph(today_df)

        today_graph.capacity_transfer(initial_graph)

        l1_norm_list.append(impression_l1(today_graph,initial_graph))
        logging_str += 'l1_norm = {}'.format(l1_norm_list)

        print(logging_str)

        ratio = [ today_graph.G(ratio=True,mode=arriving_order) for _ in range(num_loop)]
        G_ratio_list.append(ratio)
        logging_str += 'G = {}'.format(G_ratio_list)

        ratio = [ today_graph.R(mode=arriving_order) for _ in range(num_loop)]
        R_ratio_list.append(ratio)
        logging_str +='R = {}'.format(R_ratio_list)

        ratio = [today_graph.PW(improved=0,mode=arriving_order,graph=initial_graph) for _ in range(num_loop)]
        PW_ratio_list.append(ratio)
        logging_str += 'PW = {}'.format(PW_ratio_list)

        ratio = [today_graph.PW(improved=1, mode=arriving_order, graph=initial_graph) for _ in range(num_loop)]
        IPW_ratio_list.append(ratio)
        logging_str += 'IPW = {}'.format(IPW_ratio_list)



        print(logging_str)


    return 0





if __name__ == '__main__':

    csv_fname_list = ['ydata_day_{}.csv'.format(day) for day in range(1, 31)]

    robustness_test(csv_fname_list,load_mode='least-degree',arriving_order='random')

    #load_mode = 'random'
    #load_mode = 'max-min'

    #arriving_order = 'Ci-descending'
    #arriving_order = 'Ci-ascending'
    #arriving_order = 'Ca-descending'
    #arriving_order = 'Ca-ascending'