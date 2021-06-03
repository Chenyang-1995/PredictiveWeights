import math
import numpy as np
import numpy.random as random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import draw


class Advertiser:
    def __init__(self,id,capacity):
        self.id = id

        self.capacity = capacity

        #its allocation
        self.alloc = 0

        #its remaining capacity, used in greedy algo (to find the neighbor with largest remaining space)
        self.remain_capacity = self.capacity-self.alloc

        #the weight for PropAlloc
        self.weight = 1.0

        #its capacity in the training instance
        #will be decided once the impressions are sampled
        self.train_capacity = 0

        self.compute_remain_indicator()


    def compute_remain_indicator(self):
        if self.remain_capacity > 0:
            self.free = 1
        else:
            self.free = 0

        if self.capacity >0:
            self.remain_ratio = self.remain_capacity / self.capacity
        else:
            self.remain_ratio = 0


    def show(self):
        print ("id: {0}, capacity = {1}, alloc = {2}".format(self.id,
                                                             self.capacity,
                                                             self.alloc))
    #Add the capacity of this advertiser
    def add_capacity(self,capacity):
        self.capacity += capacity
        self.remain_capacity += capacity
        self.compute_remain_indicator()

    #assign impressions to this advertiser without changing the remaining capacity,
    #used when we compute the advertiser weights in the training data
    def assign_impression_unconstrained(self,size=1):
        self.alloc += size

    #assign impressions, used in greedy algo and test phase of PropAlloc
    def assign_impression_constrained(self,size=1):
        self.alloc += size
        self.remain_capacity -= size

        self.compute_remain_indicator()
        return self.free

    def get_weight(self):
        return self.weight


    #update its weight according the its Allocation and the training capacity
    def update_weight(self,eps = 1.01,eps_recip = 0.99):
        flag = 0
        ratio = self.alloc/max(0.0001,self.train_capacity*eps)

        #Note that some advertisers may got capacity 0.
        #We use max(0.0001, ) to deal with this situation.
        #Here, EPS := 1+epsilon
        if self.alloc > max(0.0001,self.train_capacity*eps):
            self.weight = self.weight*eps_recip
            flag = 1

        #Return an indicator of whether this advertiser update or not and the ratio.
        return flag,ratio


    #re_initialize before performing some other algorithms
    def re_init(self):
        self.alloc = 0
        self.remain_capacity = self.capacity-self.alloc
        self.compute_remain_indicator()

    #re_initialize before re-sampling training data and computing the advertiser weights
    def re_init_train(self):
        self.weight = 1.0
        self.train_capacity = 0

    def scale(self,factor):
        self.capacity = self.capacity//factor + 1
        self.re_init()


class Impression:
    def __init__(self,id,size,neighbour):
        self.id = id

        #the number of impressions in this type
        self.size = size

        #Used in sampling impressions to record how many impressions have not arrived.
        self.remain_size = size

        #Used in training weights phase to record how many impressions are sampled.
        self.train_size = 0

        #a list about the advertiser id adjacent to it.
        self.neighbour = neighbour

        #The key list of this dict is the same as self.neighbour,
        # the value corresponding to a key is the number of this type impressions assigned to that account
        #Used in constructing the training capacity of an advertiser
        self.opt_neighbour = {}

        #a copy of original setting, used in re_init_train()
        self.opt_neighbour_copy = {}

    def add_size(self,size):
        self.size += size
        self.remain_size += size


    #If adding edges from Rank 3, flow = then number of impressions
    #If adding edges from other Rank, flow = 0
    def add_neighbour(self,account_id,flow=0,flag=False):
        self.add_size(flow)
        if account_id not in self.opt_neighbour.keys():
            if account_id not in self.neighbour:
                self.neighbour.append(account_id)
            if flow > 0 or flag == True:
                self.opt_neighbour[account_id] = flow
                self.opt_neighbour_copy[account_id] = flow

            return 1
        else:
            if flow >0 or flag == True:
                self.opt_neighbour[account_id] += flow
                self.opt_neighbour_copy[account_id] += flow

            return 0


    #the input Ad_dict is a dict from account id to its "Advertiser" object
    #return the neighbour with the largest remaining capacity
    #If no free space, return -1,-1
    def greedy_neighbour(self,Ad_dict,Ratio=False):
        greedy_account_id = -1
        greedy_capacity = -1
        greedy_ratio = -1
        if Ratio == False:
            for account_id in self.neighbour:

                if Ad_dict[account_id].remain_capacity>0 and Ad_dict[account_id].remain_capacity>greedy_capacity:
                    greedy_account_id = account_id
                    greedy_capacity = Ad_dict[account_id].remain_capacity

            return greedy_account_id, greedy_capacity
        else:
            for account_id in self.neighbour:
                if Ad_dict[account_id].remain_capacity>0 and Ad_dict[account_id].remain_ratio>greedy_ratio:
                    greedy_account_id = account_id
                    greedy_ratio = Ad_dict[account_id].remain_ratio
            return greedy_account_id,greedy_ratio



    #assign this impression proportionally
    def proportional_assign(self,Ad_dict,flow):
        if self.remain_size >=0.5:
            self.remain_size -= flow
            weight_list = np.array([ Ad_dict[account_id].get_weight() for account_id in self.neighbour])
            if weight_list.sum() >0 :
                weight_list /= weight_list.sum()

                for index,account_id in enumerate(self.neighbour):
                    Ad_dict[account_id].assign_impression_unconstrained(size=weight_list[index]*flow)

    #assign this impression proportionally in more clever way.
    #the weights of each advertiser = its orginal weight * its free (an indicator)
    def proportional_assign_improved(self,Ad_dict,flow):
        if self.remain_size >=0.5:
            self.remain_size -= flow
            while flow > 0:
                remain_capacity_list = [ Ad_dict[ad_id].remain_capacity for ad_id in self.neighbour]
                weight_list = np.array(
                    [Ad_dict[account_id].get_weight() * Ad_dict[account_id].free for account_id in self.neighbour])
                if weight_list.sum() == 0:
                    return 0
                #if weight_list.sum() > 0:
                weight_list /= weight_list.sum()
                tmp_flow_list = []
                for index in range(len(self.neighbour)):
                    if weight_list[index] > 0:
                        tmp_flow_list.append(1.0*remain_capacity_list[index]/weight_list[index])
                    else:
                        tmp_flow_list.append(flow)

                tmp_flow = min(min(tmp_flow_list),flow)
                for index, account_id in enumerate(self.neighbour):
                    Ad_dict[account_id].assign_impression_constrained(size=weight_list[index]*tmp_flow)

                flow -= tmp_flow



    #assign this type of impressions when training weights.
    #since this is an offline setting, we assign all training size based on proportional weights directly.
    def proportional_assign_train(self,Ad_dict):

        weight_list = np.array([ Ad_dict[account_id].get_weight() for account_id in self.neighbour])
        weight_list /= weight_list.sum()
        for index,account_id in enumerate(self.neighbour):
            tmp_size = weight_list[index]*self.train_size
            Ad_dict[account_id].assign_impression_unconstrained(size=tmp_size)

    #assign an impression greedily.
    def greedy_assign(self,Ad_dict,Ratio=False,size=1):
        if self.remain_size >=0.5:
            self.remain_size -= size
            greedy_account_id, _ = self.greedy_neighbour(Ad_dict,Ratio)
            if greedy_account_id !=-1:
                Ad_dict[greedy_account_id].assign_impression_constrained(size)


    def rank_assign(self,Ad_dict,Ad_Rank_dict):

        if self.remain_size >= 0.5:
            self.remain_size -= 1
            Ad_Rank_list = [Ad_Rank_dict[account_id] for account_id in self.neighbour]

            Ranked_indices = np.argsort(-np.array(Ad_Rank_list)).tolist()

            for ranked_index in Ranked_indices:
                tmp_account_id = self.neighbour[ranked_index]

                if Ad_dict[tmp_account_id].remain_capacity > 0:
                    Ad_dict[tmp_account_id].assign_impression_constrained()
                    break



    #initialize remain_size for a new sampling phase
    def re_init(self):
        self.remain_size = self.size

    #initialize train_size and opt_neighbour for a new weight-training phase
    def re_init_train(self):
        self.train_size = 0
        self.opt_neighbour = self.opt_neighbour_copy.copy()

    def scale(self,factor):
        scaled_size = 0

        for account_id in self.opt_neighbour.keys():
            self.opt_neighbour[account_id] = self.opt_neighbour[account_id] // factor
            self.opt_neighbour_copy[account_id] = self.opt_neighbour_copy[account_id] // factor
            scaled_size += self.opt_neighbour[account_id]

        self.size = scaled_size
        self.re_init()
        self.re_init_train()

    def decrease(self):
        self.size -= 1
        non_zero_list = [account_id for account_id in self.opt_neighbour.keys() for _ in range(int(self.opt_neighbour[account_id] ))]
        account_id = np.random.choice(non_zero_list)
        self.opt_neighbour[account_id] -= 1
        self.opt_neighbour_copy[account_id] -= 1

        return account_id

    def proportional_reorganize_capacity(self,Ad_dict):
        weight_list = np.array([Ad_dict[account_id].get_weight() for account_id in self.neighbour],dtype=np.float)
        if weight_list.sum() > 0:
            weight_list /= weight_list.sum()
            weight_list *= self.size

            for index, account_id in enumerate(self.neighbour):
                Ad_dict[account_id].add_capacity(weight_list[index])
                self.opt_neighbour[account_id] = weight_list[index]
                self.opt_neighbour_copy[account_id] = weight_list[index]