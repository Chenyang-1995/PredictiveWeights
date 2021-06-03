import math
import numpy as np
import numpy.random as random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt






def draw_algos(x_list,mean_list,std_list,Algo_label,fname='Performance.jpg',x_label='Training Ratio {}'.format(chr(963)),y_label = 'Competitive Ratio'):
    plt.cla()
    fname = 'figures/{}'.format(fname)
    color_list = ['k','r', 'm', 'y', 'g', 'b', 'c','#FF00FF','#CEFFCE','#D2691E']
    marker_list = ['o', 'v', '^', '<', '>', 's','3','8','|','x']
    plt.xlabel(x_label)
    plt.ylabel(y_label)


    for i in range(len(Algo_label)):
        tmp_algo = mean_list[i]
        tmp_std = std_list[i]

        plt.errorbar(x_list, tmp_algo, yerr=0.2*np.array(tmp_std), ecolor=color_list[i], fmt='none')
        plt.plot(x_list, tmp_algo, color=color_list[i],linestyle='-', linewidth=1,label = Algo_label[i])


    if len(Algo_label) > 1:
        plt.legend(loc='lower right') #bbox_to_anchor=(0.2, 0.95))
    plt.savefig(fname, dpi=200)

