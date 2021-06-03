from graph import Graph
from vertex import Advertiser,Impression
import random
import numpy as np

class Graph_Union:
    def __init__(self,graph_list):
        self.graph_list = graph_list
        self.Ad_id_list = []

        for graph in self.graph_list:
            self.Ad_id_list += graph.Ad_id_list

        self.Ad_id_list = list(set(self.Ad_id_list))

        self.Ad_dict = {}
        for ad_id in self.Ad_id_list:
            self.Ad_dict[ad_id] = Advertiser(ad_id,0)

        for graph in self.graph_list:
            for ad_id in graph.Ad_id_list:
                self.Ad_dict[ad_id].add_capacity(graph.Ad_dict[ad_id].capacity)

        self.Im_id_list_list = [ graph.Im_id_list for graph in self.graph_list ]

        self.uniform_impressions_list_list = [ graph.uniform_impressions_list for graph in self.graph_list ]


    def sample_im(self,iter_id1,iter_id2,mode='day-order'):

        if mode == 'day-order':
            if iter_id1 == 0 and iter_id2 == 0:
                for uniform_impressions_list in self.uniform_impressions_list_list:
                    random.shuffle(uniform_impressions_list)

        return self.uniform_impressions_list_list[iter_id1][iter_id2]


    def vertex_init(self):
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].re_init()

        for graph in self.graph_list:
            for im_id in graph.Im_id_list:
                graph.Im_dict[im_id].re_init()

    def train_init(self):
        for account_id in self.Ad_id_list:
            self.Ad_dict[account_id].re_init()
            self.Ad_dict[account_id].re_init_train()
        for graph in self.graph_list:
            graph.train_init()

    def sample_im_train(self,train_ratio=0.01):
        self.train_init()

        for graph in self.graph_list:
            graph.sample_im_train(train_ratio=train_ratio)

            for ad_id in graph.Ad_id_list:
                self.Ad_dict[ad_id].train_capacity += graph.Ad_dict[ad_id].train_capacity

    def G(self, ratio=True, mode='day-order'):
        self.greedy_ratio = 0
        print('---------Start Greedy Algo with arriving order {}-----------'.format(mode))
        self.vertex_init()
        iter_id = 0
        monitor_alloc = 0
        for iter_id1,graph in enumerate(self.graph_list):
            for iter_id2 in range(len(graph.uniform_impressions_list)):
                selected_ims = self.sample_im(iter_id1,iter_id2, mode)
                for selected_im in selected_ims:
                    graph.Im_dict[selected_im].greedy_assign(self.Ad_dict,ratio)

                if iter_id2 % 10000 == 0:
                    print("iter_id2 = {0}/{1}".format(iter_id2, len(graph.uniform_impressions_list)))
                    self.show_alloc()
            print('-'*80)
            print("iter_id1 = {0}/{1}".format(iter_id1, len(self.graph_list)))
            self.show_alloc()

        print('---------Finish Greedy Algo with arriving order {}-----------'.format(mode))

        return self.Ratio

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

    def show_graph(self):
        self.sort_ad()
        self.num_edges = 0
        for graph in self.graph_list:
            for im_id in graph.Im_id_list:
                self.num_edges += len(graph.Im_dict[im_id].neighbour)

        logging_str = ''
        logging_str += 'Num edges = {}\n'.format(self.num_edges)
        logging_str += 'Uniform Impression list = {}\n'.format(sum( [len(uniform_impressions_list) for uniform_impressions_list in self.uniform_impressions_list_list ]))
        logging_str += 'Total Ad capacity = {0} with num = {1} \n'.format(self.Ad_capacity,self.Ad_num)
        #logging_str += 'Total Im size = {0} with num = {1} \n'.format(self.Im_size,self.Im_num)

        print(logging_str)

    def show_alloc(self,train=False):
        self.show_graph()
        self.Alloc = 0
        if train == False:
            for ad_id in self.Ad_id_list:
                self.Alloc += min(self.Ad_dict[ad_id].capacity,self.Ad_dict[ad_id].alloc)
            self.Ratio = 1.0*self.Alloc/self.Ad_capacity
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


    def train_ad_weights(self,train_ratio=0.01,num_iterations = 3000//500):
        print("Start training advertiser weights with train ratio {}".format(train_ratio))
        self.train_init()

        self.sample_im_train(train_ratio=train_ratio)

        #flag is used to record how many advertisers updated in an iteration
        flag = 1

        #record the number of iterations
        iter = 0
        count_max_ratio = 0
        last_max_ratio = -100
        while flag >0 and count_max_ratio <= num_iterations:
            flag = 0
            max_ratio = 0
            sum_ratio = 0
            for graph in self.graph_list:
                for im_id in graph.Im_id_list:
                    graph.Im_dict[im_id].proportional_assign_train(self.Ad_dict)

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
            for account_id in self.Ad_id_list:
                self.Ad_dict[account_id].re_init()

            iter += 1

    def PW(self,improved, mode='day-order', train_ratio=0.01,graph=None):
        print('-'*80)
        if graph == None:

            self.train_ad_weights(train_ratio)

        elif graph == 'Save':
            pass
        else:
            pass#self.ad_weight_transfer(graph)

        self.vertex_init()
        print('-'*80)
        print("Start Proportional Weights Algo with train ratio {0}, arriving order {1}, improved = {2}".format(train_ratio, mode,improved))

        for iter_id1,graph in enumerate(self.graph_list):
            for iter_id2 in range(len(graph.uniform_impressions_list)):
                selected_ims = self.sample_im(iter_id1, iter_id2, mode)
                if improved == 0:
                    graph.Im_dict[selected_ims[0]].proportional_assign(self.Ad_dict, len(selected_ims))
                elif improved == 1:
                    graph.Im_dict[selected_ims[0]].proportional_assign_improved(self.Ad_dict, len(selected_ims))

                if iter_id2 % 10000 == 0:
                    print("iter_id2 = {0}/{1}".format(iter_id2, len(graph.uniform_impressions_list)))

                    self.show_alloc()

            print('-'*80)
            print("iter_id1 = {0}/{1}".format(iter_id1, len(self.graph_list)))
            self.show_alloc()

        print("Finish Proportional Weights Algo with train ratio {0}, arriving order {1}, improved = {2}".format(train_ratio, mode,improved))
        print('-'*80)

        return self.Ratio

    def R(self,mode = 'day-order'):

        random.shuffle(self.Ad_id_list)
        Ad_Rank_dict = {}

        for index, account_id in enumerate(self.Ad_id_list):
            Ad_Rank_dict[account_id] = index

        self.vertex_init()
        print('---------Start Rank Algo with arriving order {}-----------'.format(mode))

        for iter_id1,graph in enumerate(self.graph_list):
            for iter_id2 in range(len(graph.uniform_impressions_list)):
                selected_ims = self.sample_im(iter_id1,iter_id2, mode)
                for selected_im in selected_ims:
                    graph.Im_dict[selected_im].rank_assign(self.Ad_dict, Ad_Rank_dict)


                if iter_id2 % 10000 == 0:
                    print("iter_id2 = {0}/{1}".format(iter_id2, len(graph.uniform_impressions_list)))
                    self.show_alloc()
            print('-'*80)
            print("iter_id1 = {0}/{1}".format(iter_id1, len(self.graph_list)))
            self.show_alloc()


        print('---------Finish Rank Algo with arriving order {}-----------'.format(mode))
        return self.Ratio

def impression_l1(graph1,graph2):
        tmp_im_id_list = graph1.Im_id_list + graph2.Im_id_list
        tmp_im_id_list = list(set(tmp_im_id_list))

        l1_norm = 0
        for im_id in tmp_im_id_list:
            if im_id in graph1.Im_id_list and im_id in graph2.Im_id_list:
                difference = 1.0*graph1.Im_dict[im_id].size/graph1.Im_size - 1.0*graph2.Im_dict[im_id].size/graph2.Im_size
                l1_norm += abs(difference)
            elif im_id in graph1.Im_id_list:
                l1_norm += 1.0*graph1.Im_dict[im_id].size/graph1.Im_size
            elif im_id in graph2.Im_id_list:
                l1_norm += 1.0*graph2.Im_dict[im_id].size/graph2.Im_size

        return l1_norm


def phrase_l1(graph1, graph2):


    keyphrase_list = graph1.key_phrase_list + graph2.key_phrase_list
    keyphrase_list = list(set(keyphrase_list))

    sum_graph1_keyphrases = sum([ len(graph1.key_phrase_counter[k]) for k in graph1.key_phrase_list ])

    sum_graph2_keyphrases = sum([len(graph2.key_phrase_counter[k]) for k in graph2.key_phrase_list])
    l1_norm = 0

    for k in keyphrase_list:
        if k in graph1.key_phrase_list and k in graph2.key_phrase_list:
            difference = 1.0*len(graph1.key_phrase_counter[k])/sum_graph1_keyphrases - 1.0*len(graph2.key_phrase_counter[k])/sum_graph2_keyphrases
            l1_norm += abs(difference)
        elif k in graph1.key_phrase_list:
            l1_norm += 1.0*len(graph1.key_phrase_counter[k])/sum_graph1_keyphrases
        elif k in graph2.key_phrase_list:
            l1_norm += 1.0*len(graph2.key_phrase_counter[k])/sum_graph2_keyphrases


    return l1_norm


def capacity_l1(graph1, graph2):
    ad_id_list = graph1.Ad_id_list + graph2.Ad_id_list
    ad_id_list = list(set(ad_id_list))

    sum_graph1_capacity = sum([graph1.Ad_dict[ad_id].capacity for ad_id in graph1.Ad_id_list])

    sum_graph2_capacity = sum([graph2.Ad_dict[ad_id].capacity for ad_id in graph2.Ad_id_list])
    l1_norm = 0

    for ad_id in ad_id_list:
        if ad_id in graph1.Ad_id_list and ad_id in graph2.Ad_id_list:
            difference = 1.0 * graph1.Ad_dict[ad_id].capacity / sum_graph1_capacity - 1.0 * graph2.Ad_dict[ad_id].capacity / sum_graph2_capacity
            l1_norm += abs(difference)
        elif ad_id in graph1.Ad_id_list:
            l1_norm += 1.0 * graph1.Ad_dict[ad_id].capacity / sum_graph1_capacity
        elif ad_id in graph2.Ad_id_list:
            l1_norm += 1.0 * graph2.Ad_dict[ad_id].capacity / sum_graph2_capacity

    return l1_norm



