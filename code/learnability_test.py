import math
import numpy as np
import numpy.random as random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt

from graph import *
from graph_union import *
from data_preprocess import *




def learnability_test(csv_fname_list,load_mode='least-degree',arriving_order='random'):
    print("Start {}!".format(csv_fname_list))


    if arriving_order != 'day-order':
        df_list = []
        G_ratio = []
        R_ratio = []
        train_ratios = [0.001,0.003,0.009,0.2,0.027,0.4,0.6,0.8]
        PW_ratio = {}
        IPW_ratio = {}
        for tr in train_ratios:
            PW_ratio[tr] = []
            IPW_ratio[tr] = []
        for i in range(len(csv_fname_list)):
            logging_str = ''
            print('--------Day {}-----------'.format(i+1))
            logging_str += '--------Day {}-----------\n'.format(i+1)
            today_df = read_data(csv_fname_list[i])

            today_graph = Graph(today_df,load_mode=load_mode)

            num_loop = 4
            ratio = [today_graph.G(mode=arriving_order) for _ in range(num_loop)]
            G_ratio.append(ratio)
            logging_str += 'G = {}\n'.format(G_ratio)

            ratio = [today_graph.R(mode=arriving_order) for _ in range(num_loop)]
            R_ratio.append(ratio)
            logging_str += 'R = {}\n'.format(R_ratio)


            for tr in train_ratios:
                ratio0 = []
                ratio1 = []
                for _ in range(num_loop):
                    ratio0.append(today_graph.PW(improved=0,train_ratio=0.01,mode=arriving_order))

                    ratio1.append(today_graph.PW(improved=1, train_ratio=0.01, mode=arriving_order, graph='Save'))

                PW_ratio[tr].append(ratio0)
                IPW_ratio[tr].append(ratio1)
                logging_str += 'PW_{0} = {1}\n'.format(tr,PW_ratio[tr])
                logging_str += 'IPW_{0} = {1}\n'.format(tr,IPW_ratio[tr])


            print(logging_str)

        return 0
    else:
        df_list = [read_data(csv_fname) for csv_fname in csv_fname_list[:7]]
        graph_list = [ Graph(df) for df in df_list ]
        graph_union = Graph_Union(graph_list)

        G_ratio = []
        R_ratio = []
        train_ratios = [0.001,0.003,0.009,0.2,0.027,0.4,0.6,0.8]
        PW_ratio = {}
        IPW_ratio = {}
        for tr in train_ratios:
            PW_ratio[tr] = []
            IPW_ratio[tr] = []

        num_loop = 4
        logging_str = ''

        G_ratio = [graph_union.G(mode=arriving_order) for _ in range(num_loop)]

        logging_str += 'G = {}\n'.format(G_ratio)

        R_ratio = [graph_union.R(mode=arriving_order) for _ in range(num_loop)]

        logging_str += 'R = {}\n'.format(R_ratio)

        for tr in train_ratios:
            for _ in range(num_loop):
                PW_ratio[tr].append(graph_union.PW(improved=0, train_ratio=tr, mode=arriving_order))

                IPW_ratio[tr].append(graph_union.PW(improved=1, train_ratio=tr, mode=arriving_order, graph='Save'))

                logging_str += 'PW_{0} = {1}\n'.format(tr,PW_ratio[tr])
                logging_str += 'IPW_{0} = {1}\n'.format(tr,IPW_ratio[tr])


        return 0



if __name__ == '__main__':
    csv_fname_list = ['ydata_day_{}.csv'.format(day) for day in range(1, 31)]

    learnability_test(csv_fname_list,load_mode='least-degree',arriving_order='random')
    #load_mode = 'random'
    #load_mode = 'max-min'

    #arriving_order = 'Ci-descending'
    #arriving_order = 'Ci-ascending'
    #arriving_order = 'Ca-descending'
    #arriving_order = 'Ca-ascending'
    #arriving_order = 'day-order'
