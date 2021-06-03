import math
import numpy as np
import numpy.random as random
import mmap
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain, combinations
import draw


from vertex import Impression,Advertiser

def powerset(iterable):

    s = list(iterable).copy()
    #s.sort()
    return list(chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1)))

class Graph:
    def __init__(self,data,load_mode = "least-degree",scale_factor = None):
        self.data = data

        # the set of Ranks
        self.Rank_list = data['Rank']
        self.Rank_list = self.Rank_list.unique()
        self.Rank_list.sort()

        # find the rank with the maximum number of impressions
        # for day 1, the value is Rank 3.
        self.max_num_per_rank = -1
        self.max_rank = -1
        for rank in self.Rank_list:
            tmp_data = data[data['Rank'] == rank]
            tmp_num = tmp_data['Impressions'].sum()
            if tmp_num > self.max_num_per_rank:
                self.max_num_per_rank = tmp_num
                self.max_rank = rank

        # select the data of Rank 3
        self.data_rank = data[data['Rank'] == self.max_rank]
        # self.data_rank = data[data['Rank'] <= 5]
        # self.data_rank = data

        self.num_row = self.data_rank.shape[0]

        # the account id list of Rank 3
        self.Ad_id_list = self.data_rank['Account_id']
        self.Ad_id_list = self.Ad_id_list.unique()
        # self.Ad_id_list = self.Ad_id_list[100:200]

        # the impression id list of Rank3
        self.Im_id_list = self.data_rank['Keyphrase']
        self.Im_id_list = self.Im_id_list.unique()

        print(self.Im_id_list[0])

        self.load_mode = load_mode
        #least degree
        if load_mode == "least-degree":
            self.load_data_least_degree()
        elif load_mode == 'random':
            #random
            self.load_data_random()
        elif load_mode == "max-min":
            #max_min
            self.load_data_max_min(scale_factor)


    def load_key_phrases(self):
        self.key_phrase_counter = defaultdict(list)

        for index in range(self.num_row):
            if index % 10000 == 0:
                print("finish load key_phrases {0}/{1}".format(index + 1, self.num_row))
            tmp_data = self.data_rank.iloc[index]
            tmp_account_id = tmp_data['Account_id']
            if tmp_data['Account_id'] not in self.Ad_id_list:
                continue
            tmp_impression_id = tmp_data['Keyphrase']#+'-'+str(tmp_data['Avg_bid'])#.map(str)

            tmp_num =  int(tmp_data['Impressions'])
            tmp_capacity = tmp_num
            key_phrase_list = tmp_data['Keyphrase'].split('-')

            for key_phrase in key_phrase_list:
                self.key_phrase_counter[key_phrase] += [tmp_impression_id for _ in range(tmp_num)]




    def sort_key_phrases(self):
        logging_str = ''
        self.key_phrase_list = [ key_phrase for key_phrase in self.key_phrase_counter.keys() ]
        logging_str += 'key_phrases_num = {}'.format(len(self.key_phrase_list))
        self.key_phrase_lens = [ len(self.key_phrase_counter[key_phrase]) for key_phrase in self.key_phrase_list ]
        tmp_indices = np.argsort(-np.array(self.key_phrase_lens)).tolist()
        self.key_phrase_lens = [self.key_phrase_lens[index] for index in tmp_indices]
        self.key_phrase_list = [ self.key_phrase_list[index] for index in tmp_indices ]
        logging_str += ' Top key phrases len = {}'.format(self.key_phrase_lens[:min(10,len(self.key_phrase_list))])

        print (logging_str)

    def sort_ad(self,degree = False):
        self.Ad_id_list = list(set(self.Ad_id_list))
        self.Ad_num = len(self.Ad_id_list)
        self.Ad_capacity_list = [self.Ad_dict[ad_id].capacity for ad_id in self.Ad_id_list]
        if degree == False:
            tmp_indices = np.argsort(-np.array(self.Ad_capacity_list)).tolist()
            self.Ad_capacity = sum(self.Ad_capacity_list)
            self.Ad_capacity_list = [self.Ad_capacity_list[index] for index in tmp_indices]
            self.Ad_id_list = [self.Ad_id_list[index] for index in tmp_indices]
        else:
            tmp_indices = np.argsort(-np.array(self.Ad_degree_list)).tolist()
            self.Ad_capacity = sum(self.Ad_capacity_list)
            self.Ad_capacity_list = [self.Ad_capacity_list[index] for index in tmp_indices]
            self.Ad_id_list = [self.Ad_id_list[index] for index in tmp_indices]
            self.Ad_degree_list = [self.Ad_degree_list[index] for index in tmp_indices ]

    def sort_im(self,LEN=False):

        self.Im_id_list = list(set(self.Im_id_list))

        self.Im_num = len(self.Im_id_list)
        self.Im_size_list = [self.Im_dict[im_id].size for im_id in self.Im_id_list]
        self.Im_size = sum(self.Im_size_list)
        self.Im_len_list = [ len(im_id) for im_id in self.Im_id_list]
        if LEN == False:
            tmp_indices = np.argsort(-np.array(self.Im_size_list)).tolist()

        else:
            tmp_indices = np.argsort(np.array(self.Im_len_list)).tolist()

        self.Im_size_list = [self.Im_size_list[index] for index in tmp_indices]
        self.Im_id_list = [self.Im_id_list[index] for index in tmp_indices]
        self.Im_len_list = [self.Im_len_list[index] for index in tmp_indices]


    def filter_key_phrases(self,Bottom=0,Top=20):
        self.sort_key_phrases()
        self.key_phrase_list = self.key_phrase_list[Bottom:Top]
        self.key_phrase_lens = self.key_phrase_lens[Bottom:Top]
        print('selected_key_phrases = {}'.format(self.key_phrase_lens))
        self.Im_id_list = powerset(self.key_phrase_list)

    def load_data_least_degree(self):
        self.load_key_phrases()
        self.filter_key_phrases()
        self.Im_dict = {}
        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id] = Impression(impression_id,0,[])

        self.Ad_dict = {}
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id] = Advertiser(account_id,0)

        self.uniform_impressions_list = []
        self.Ad_id_list = []

        self.Ad_neighbour = defaultdict(list)

        print('key phrase = {}'.format(self.key_phrase_list))
        for index in range(self.num_row):
            if index % 10000 == 0:
                print("finish load data {0}/{1}".format(index + 1, self.num_row))
            tmp_data = self.data_rank.iloc[index]
            tmp_account_id = tmp_data['Account_id']

            tmp_impression_id = tmp_data['Keyphrase']

            tmp_num =  int(tmp_data['Impressions'])
            tmp_capacity = tmp_num
            tmp_key_phrase_list = tmp_data['Keyphrase'].split('-')


            feasible_key_phrases = [key_phrase for key_phrase in tmp_key_phrase_list if key_phrase in self.key_phrase_list]
            feasible_key_phrases = list(set(feasible_key_phrases))
            feasible_key_phrases.sort(key=lambda x:self.key_phrase_list.index(x))


            if len(feasible_key_phrases) >0:
                feasible_power_set = powerset(feasible_key_phrases)

                mean_size = tmp_num // len(feasible_power_set)
                tmp_num = mean_size*len(feasible_power_set)

                if mean_size > 0:

                    tmp_capacity = tmp_num / len(feasible_key_phrases)

                    self.Ad_id_list.append(tmp_account_id)

                    for i,impression_id in enumerate(feasible_power_set):

                        if len(impression_id) == 1:
                            pass
                        if i < len(feasible_power_set) - 1:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id,0)# mean_size)
                            self.Ad_neighbour[tmp_account_id].append(impression_id)


                        else:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id, tmp_num)
                            self.Ad_neighbour[tmp_account_id].append( impression_id )
                            self.uniform_impressions_list.append( [impression_id for _ in range(tmp_num)] )

        self.Im_id_list = [ im_id for im_id in self.Im_id_list if self.Im_dict[im_id].size >0 ]


        self.sort_im()
        self.sort_ad()
        self.Ad_degree_list = []
        self.Ad_degree_dict = {}
        for ad_id in self.Ad_id_list:
            self.Ad_neighbour[ad_id] = list(set(self.Ad_neighbour[ad_id]))
            self.Ad_degree_list.append(len(self.Ad_neighbour[ad_id]))
            self.Ad_degree_dict[ad_id] = len(self.Ad_neighbour[ad_id])

        marked_ad = defaultdict(int)
        self.sort_im()

        for im_id in self.Im_id_list:
            tmp_degree_list = [self.Ad_degree_dict[ad_id] for ad_id in self.Im_dict[im_id].neighbour]

            least_degree = min(tmp_degree_list)
            tmp_ad_list = [ ad_id for ad_id in self.Im_dict[im_id].neighbour if self.Ad_degree_dict[ad_id]==least_degree]

            mean_size = 1.0*self.Im_dict[im_id].size / len(tmp_ad_list)
            for ad_id in tmp_ad_list:
                self.Ad_dict[ad_id].add_capacity(mean_size)


        self.show_graph()

    def load_data_random(self):
        self.load_key_phrases()
        self.filter_key_phrases()
        self.Im_dict = {}
        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id] = Impression(impression_id,0,[])

        self.Ad_dict = {}
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id] = Advertiser(account_id,0)

        self.uniform_impressions_list = []
        self.Ad_id_list = []

        self.Ad_neighbour = defaultdict(list)

        print('key phrase = {}'.format(self.key_phrase_list))
        for index in range(self.num_row):
            if index % 10000 == 0:
                print("finish load data {0}/{1}".format(index + 1, self.num_row))
            tmp_data = self.data_rank.iloc[index]
            tmp_account_id = tmp_data['Account_id']

            tmp_impression_id = tmp_data['Keyphrase']#+'-'+str(tmp_data['Avg_bid'])#.map(str)

            tmp_num =  int(tmp_data['Impressions'])
            tmp_capacity = tmp_num
            tmp_key_phrase_list = tmp_data['Keyphrase'].split('-')


            feasible_key_phrases = [key_phrase for key_phrase in tmp_key_phrase_list if key_phrase in self.key_phrase_list]
            feasible_key_phrases = list(set(feasible_key_phrases))
            feasible_key_phrases.sort(key=lambda x:self.key_phrase_list.index(x))


            if len(feasible_key_phrases) >0:
                feasible_power_set = powerset(feasible_key_phrases)

                mean_size = tmp_num // len(feasible_power_set)
                tmp_num = mean_size*len(feasible_power_set)

                if mean_size > 0:# and tmp_num<=1000:

                    tmp_capacity = tmp_num / len(feasible_key_phrases)

                    self.Ad_id_list.append(tmp_account_id)

                    for i,impression_id in enumerate(feasible_power_set):


                        if len(impression_id) == 1:
                            pass

                        if i < len(feasible_power_set) - 1:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id,0)
                            self.Ad_neighbour[tmp_account_id].append(impression_id)

                        else:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id, tmp_num)
                            self.Ad_neighbour[tmp_account_id].append( impression_id )

                            self.uniform_impressions_list.append( [impression_id for _ in range(tmp_num)] )

        self.Im_id_list = [ im_id for im_id in self.Im_id_list if self.Im_dict[im_id].size >0 ]


        self.sort_im()
        self.sort_ad()
        self.Ad_degree_list = []
        self.Ad_degree_dict = {}
        for ad_id in self.Ad_id_list:
            self.Ad_neighbour[ad_id] = list(set(self.Ad_neighbour[ad_id]))
            self.Ad_degree_list.append(len(self.Ad_neighbour[ad_id]))
            self.Ad_degree_dict[ad_id] = len(self.Ad_neighbour[ad_id])



        marked_ad = defaultdict(int)

        self.sort_im()

        self.capacity_map_dict = defaultdict(int)
        for im_id in self.Im_id_list:

            tmp_ad_list = []


            tmp_ad_list = self.Im_dict[im_id].neighbour.copy()
            mean_size = 1.0*self.Im_dict[im_id].size / len(tmp_ad_list)

            if len(tmp_ad_list) == 1:
                mean_size_list = [mean_size]
            else:

                break_point = np.random.choice([i for i in range(1,self.Im_dict[im_id].size)],size=len(tmp_ad_list)-1,replace=True)
                break_point = list(break_point)
                break_point.sort()
                break_point.append(self.Im_dict[im_id].size)
                mean_size_list = []
                mean_size_list.append(break_point[0])
                for break_index in range(1,len(break_point)):
                    mean_size_list.append(break_point[break_index]-break_point[break_index-1])

            for ad_id in tmp_ad_list:
                tmp_size = mean_size_list.pop(0)
                self.Ad_dict[ad_id].add_capacity(tmp_size)
                self.capacity_map_dict[(im_id,ad_id)] =1.0*tmp_size/self.Im_dict[im_id].size


        self.show_graph()

    def load_data_max_min(self,scale_factor):
        self.load_key_phrases()
        self.filter_key_phrases()
        self.Im_dict = {}
        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id] = Impression(impression_id,0,[])

        self.Ad_dict = {}
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id] = Advertiser(account_id,0)

        self.uniform_impressions_list = []
        self.Ad_id_list = []

        self.Ad_neighbour = defaultdict(list)

        print('key phrase = {}'.format(self.key_phrase_list))
        for index in range(self.num_row):
            if index % 10000 == 0:
                print("finish load data {0}/{1}".format(index + 1, self.num_row))
            tmp_data = self.data_rank.iloc[index]
            tmp_account_id = tmp_data['Account_id']

            tmp_impression_id = tmp_data['Keyphrase']#+'-'+str(tmp_data['Avg_bid'])#.map(str)

            tmp_num =  int(tmp_data['Impressions'])
            tmp_capacity = tmp_num
            tmp_key_phrase_list = tmp_data['Keyphrase'].split('-')


            feasible_key_phrases = [key_phrase for key_phrase in tmp_key_phrase_list if key_phrase in self.key_phrase_list]
            feasible_key_phrases = list(set(feasible_key_phrases))
            feasible_key_phrases.sort(key=lambda x:self.key_phrase_list.index(x))


            if len(feasible_key_phrases) >0:
                feasible_power_set = powerset(feasible_key_phrases)

                mean_size = tmp_num // len(feasible_power_set)
                tmp_num = mean_size*len(feasible_power_set)

                if mean_size > 0:

                    tmp_capacity = tmp_num / len(feasible_key_phrases)

                    self.Ad_id_list.append(tmp_account_id)

                    for i,impression_id in enumerate(feasible_power_set):


                        if len(impression_id) == 1:
                            pass

                        if i < len(feasible_power_set) - 1:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id,0)# mean_size)
                            self.Ad_neighbour[tmp_account_id].append(impression_id)


                        else:
                            self.Im_dict[impression_id].add_neighbour(tmp_account_id, tmp_num)

                            self.Ad_neighbour[tmp_account_id].append( impression_id )

                            self.uniform_impressions_list.append( [impression_id for _ in range(tmp_num)] )


        if scale_factor != None:
            self.uniform_impressions_list = []
            for impression_id in self.Im_id_list:
                self.Im_dict[impression_id].scale(scale_factor)
                self.uniform_impressions_list.append(
                    [impression_id for _ in range(int(self.Im_dict[impression_id].size))])

        self.Im_id_list = [ im_id for im_id in self.Im_id_list if self.Im_dict[im_id].size >0 ]



        self.sort_im()
        self.sort_ad()
        self.Ad_degree_list = []
        self.Ad_degree_dict = {}
        for ad_id in self.Ad_id_list:
            self.Ad_neighbour[ad_id] = list(set(self.Ad_neighbour[ad_id]))
            self.Ad_degree_list.append(len(self.Ad_neighbour[ad_id]))
            self.Ad_degree_dict[ad_id] = len(self.Ad_neighbour[ad_id])

        marked_ad = defaultdict(int)

        self.capacity_map_dict = defaultdict(int)
        self.Im_id_list.sort()
        for im_id in self.Im_id_list:

            for _ in range(int(self.Im_dict[im_id].size)):
                    tmp_capacity = [ self.Ad_dict[ad_id].capacity for ad_id in self.Im_dict[im_id].neighbour]

                    min_cap = min(tmp_capacity)
                    min_index = tmp_capacity.index(min_cap)

                    min_ad = self.Im_dict[im_id].neighbour[min_index]

                    self.Ad_dict[min_ad].add_capacity(1)
                    self.capacity_map_dict[(im_id,min_ad)] += 1.0/self.Im_dict[im_id].size

        self.show_graph()

    def show_graph(self):
        self.sort_ad()
        self.sort_im()
        self.num_edges = 0
        for im_id in self.Im_id_list:
            self.num_edges += len(self.Im_dict[im_id].neighbour)

        logging_str = ''
        logging_str += 'Num edges = {}\n'.format(self.num_edges)
        logging_str += 'Uniform Impression list = {}\n'.format(len(self.uniform_impressions_list))
        logging_str += 'Total Ad capacity = {0} with num = {1} \n'.format(self.Ad_capacity,self.Ad_num)
        logging_str += 'Total Im size = {0} with num = {1} \n'.format(self.Im_size,self.Im_num)



        x_list = [i for i in range(len(self.Im_id_list))]
        draw.draw_competitive_ratio_over_impression_ratio(x_list, self.Im_size_list,
                                                          fname='Large_Impression_size.jpg',
                                                          x_label='Impression_id',
                                                          y_label='Impression_size')

        x_list = [i for i in range(len(self.Ad_id_list))]
        draw.draw_competitive_ratio_over_impression_ratio(x_list, self.Ad_capacity_list,
                                                          fname='Large_Account_capacity.jpg',
                                                          x_label='Advertiser_id',
                                                          y_label='Advertiser_size')


        print(logging_str)

    def show_alloc(self,train=False):
        self.show_graph()
        self.Alloc = 0
        if train == False:
            for ad_id in self.Ad_id_list:
                self.Alloc += min(self.Ad_dict[ad_id].capacity,self.Ad_dict[ad_id].alloc)
            self.Ratio = max(1.0*self.Alloc/self.Im_size,1.0*self.Alloc/self.Ad_capacity)
        else:
            train_ad_capacity = 0
            for ad_id in self.Ad_id_list:
                self.Alloc += min(self.Ad_dict[ad_id].train_capacity, self.Ad_dict[ad_id].alloc)
                train_ad_capacity += self.Ad_dict[ad_id].train_capacity
            self.Ratio =  1.0 * self.Alloc / train_ad_capacity
        logging_str = ''
        logging_str += 'Alloc = {}, '.format(self.Alloc)
        logging_str += 'Ratio = {:<5.3f}'.format(self.Ratio)
        print(logging_str)

    def vertex_init(self):
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].re_init()

        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id].re_init()

    def train_init(self):
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].re_init()
            self.Ad_dict[account_id].re_init_train()
        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id].re_init()
            self.Im_dict[impression_id].re_init_train()

    def sample_im(self,iter_id,mode):
        if mode == "random":
            if iter_id == 0:
                random.shuffle(self.uniform_impressions_list)

        elif mode == "Ci-descending":
            if iter_id == 0:
                tmp_size_list = [ self.Im_dict[im_ids[0]].size for im_ids in self.uniform_impressions_list ]
                tmp_indices = np.argsort(-np.array(tmp_size_list)).tolist()

                self.uniform_impressions_list = [self.uniform_impressions_list[index] for index in tmp_indices]

        elif mode == "Ci-ascending":
            if iter_id == 0:
                tmp_size_list = [self.Im_dict[im_ids[0]].size for im_ids in self.uniform_impressions_list]
                tmp_indices = np.argsort(np.array(tmp_size_list)).tolist()

                self.uniform_impressions_list = [self.uniform_impressions_list[index] for index in tmp_indices]

        elif mode == "Ca-descending":
            if iter_id == 0:
                tmp_size_list = [sum([self.Ad_dict[ad_id].capacity for ad_id in self.Im_dict[im_ids[0]].neighbour]) for im_ids in self.uniform_impressions_list]
                tmp_indices = np.argsort(-np.array(tmp_size_list)).tolist()

                self.uniform_impressions_list = [self.uniform_impressions_list[index] for index in tmp_indices]
        elif mode == "Ca-ascending":
            if iter_id == 0:
                tmp_size_list = [sum([self.Ad_dict[ad_id].capacity for ad_id in self.Im_dict[im_ids[0]].neighbour]) for
                                 im_ids in self.uniform_impressions_list]
                tmp_indices = np.argsort(np.array(tmp_size_list)).tolist()

                self.uniform_impressions_list = [self.uniform_impressions_list[index] for index in tmp_indices]


        return self.uniform_impressions_list[iter_id]

    def sample_im_train(self,train_ratio):
        random.shuffle(self.uniform_impressions_list)
        tmp_len = int(len(self.uniform_impressions_list)*train_ratio)
        total_train_size = 0
        for im_ids in self.uniform_impressions_list[:tmp_len]:
            for im_id in im_ids:
                self.Im_dict[im_id].train_size += 1
                total_train_size += 1
        if train_ratio < 0.99:
            if self.load_mode == "least-degree":
                for im_id in self.Im_id_list:
                    tmp_degree_list = [self.Ad_degree_dict[ad_id] for ad_id in self.Im_dict[im_id].neighbour]

                    least_degree = min(tmp_degree_list)
                    tmp_ad_list = [ad_id for ad_id in self.Im_dict[im_id].neighbour if
                                   self.Ad_degree_dict[ad_id] == least_degree]

                    mean_size = 1.0 * self.Im_dict[im_id].train_size / len(tmp_ad_list)
                    for ad_id in tmp_ad_list:
                        self.Ad_dict[ad_id].train_capacity += mean_size
            else:
                for im_id in self.Im_id_list:
                    for ad_id in self.Im_dict[im_id].neighbour:
                        self.Ad_dict[ad_id].train_capacity += self.capacity_map_dict[(im_id,ad_id)]*self.Im_dict[im_id].train_size
        else:
            for ad_id in self.Ad_id_list:
                self.Ad_dict[ad_id].train_capacity = self.Ad_dict[ad_id].capacity


    def G(self, ratio=True, mode="random"):
        self.greedy_ratio = 0
        print('--------- Start Greedy Algo with arriving order {} -----------'.format(mode))
        self.vertex_init()
        iter_id = 0
        monitor_alloc = 0
        while iter_id < len(self.uniform_impressions_list):
            selected_ims = self.sample_im(iter_id, mode)
            for selected_im in selected_ims:
                self.Im_dict[selected_im].greedy_assign(self.Ad_dict, ratio)

            if iter_id % 10000 == 0:
                    print("iter_id = {0}/{1}".format(iter_id, len(self.uniform_impressions_list)))
                    self.show_alloc()
            iter_id += 1

        print("iter_id = {0}/{1}".format(iter_id, len(self.uniform_impressions_list)))

        self.show_alloc()
        print('---------Finish Greedy Algo with arriving order {}-----------'.format(mode))


        return self.Ratio

    def R(self,mode = 'random'):

        random.shuffle(self.Ad_id_list)
        Ad_Rank_dict = {}

        for index, account_id in enumerate(self.Ad_id_list):
            Ad_Rank_dict[account_id] = index

        self.vertex_init()
        print('---------Start Ranking Algo with arriving order {}-----------'.format(mode))
        for iter_id in range(len(self.uniform_impressions_list)):
            selected_impression_ids = self.sample_im(iter_id, mode)
            for selected_impression_id in selected_impression_ids:
                self.Im_dict[selected_impression_id].rank_assign(self.Ad_dict, Ad_Rank_dict)

            if iter_id % 10000 == 0:
                    print("iter_id = {0}/{1}".format(iter_id, len(self.uniform_impressions_list)))


        self.show_alloc()
        print('---------Finish Ranking Algo with arriving order {}-----------'.format(mode))
        return self.Ratio

    def train_ad_weights(self,train_ratio=0.01,num_iterations = 3000//500):
        print("Start training advertiser weights with train ratio {}".format(train_ratio))
        start_time = time.time()
        self.train_init()

        self.sample_im_train(train_ratio=train_ratio)

        #flag is used to record how many advertisers updated in an iteration
        flag = 1

        #record the number of iterations
        iter = 0
        count_max_ratio = 0
        last_max_ratio = -100
        while flag >0 and count_max_ratio <= num_iterations:
            for account_id in self.Ad_id_list:
                self.Ad_dict[account_id].re_init()
            flag = 0
            max_ratio = 0
            sum_ratio = 0
            for impression_id in self.Im_id_list:
                self.Im_dict[impression_id].proportional_assign_train(self.Ad_dict)
            for account_id in self.Ad_id_list:
                tmp_flag,tmp_ratio = self.Ad_dict[account_id].update_weight()
                flag += tmp_flag
                max_ratio = max(max_ratio,tmp_ratio)
                sum_ratio += tmp_ratio

            if iter % 500 == 0 or flag == 0:
                print ("iter = {0}, number of updated advertisers = {1}, ".format(iter,flag) )
                print ("max_ratio = {:<10.2f}".format(max_ratio))
                print("count_max_ratio = {}".format(count_max_ratio))
                self.show_alloc(train=True)

                if last_max_ratio - self.Ratio < 0.00001 and last_max_ratio - self.Ratio > -0.00001:
                    count_max_ratio += 1
                else:

                    # self.show_alloc(train=True)
                    count_max_ratio = 0
                    last_max_ratio = self.Ratio


            iter += 1

        self.show_alloc(train=True)
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].re_init()

        end_time = time.time()
        print("Finish training advertiser weights with train ratio {0}, time = {1} min".format(train_ratio,(end_time-start_time)//60))
        return self.Ratio

    def PW(self,improved, mode='random', train_ratio=0.01,graph=None):
        print('-'*80)
        if graph == None:

            self.train_ad_weights(train_ratio)

        elif graph == 'Save':
            pass
        else:
            self.ad_weight_transfer(graph)

        self.vertex_init()
        print(
            "Start Proportional Weights Algo with train ratio {0}, arriving order {1}, improved = {2}".format(
                train_ratio, mode,improved))

        iter_id = 0
        while iter_id < len(self.uniform_impressions_list):
            selected_ims = self.sample_im(iter_id, mode)
            if improved == 0:
                self.Im_dict[selected_ims[0]].proportional_assign(self.Ad_dict,len(selected_ims))
            elif improved == 1:
                self.Im_dict[selected_ims[0]].proportional_assign_improved(self.Ad_dict,len(selected_ims))

            if iter_id % 10000 == 0:
                print("iter_id = {0}/{1}".format(iter_id, len(self.uniform_impressions_list)))

                self.show_alloc()

            iter_id += 1
        self.show_alloc()
        print('+'*80)
        return self.Ratio


    def scale(self,factor):
        self.uniform_impressions_list = []
        for impression_id in self.Im_id_list:
            self.Im_dict[impression_id].scale(factor)
            self.uniform_impressions_list.append([impression_id for _ in range(int(self.Im_dict[impression_id].size))])


        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].scale(factor)

        self.show_graph()

    def ad_weight_transfer(self,graph):
        self.train_init()
        for account_id in graph.Ad_id_list:
            if account_id in self.Ad_id_list:
                self.Ad_dict[account_id].weight = graph.Ad_dict[account_id].weight

    def capacity_transfer(self,graph):
        self.ad_weight_transfer(graph)

        for account_id in graph.Ad_id_list:
            if account_id in self.Ad_id_list:
                self.Ad_dict[account_id].capacity = graph.Ad_dict[account_id].capacity
