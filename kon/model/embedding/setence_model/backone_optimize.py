# _*_ coding:utf-8 _*_
import numpy as np
from numpy import random

class optimize_funcation():

    def __init__(self):
        pass

    def generate_alias_table(self, all_probability):
        num_probability=len(all_probability)

        all_probability=list((np.array(all_probability)*num_probability)/np.sum(all_probability))


        small, large = [], []
        prab, alias = [-1] * num_probability, [-1] * num_probability

        format_count=0
        for prob_rank in range(num_probability):
            if all_probability[prob_rank] == 1:
                prab[prob_rank] = 1
                alias[prob_rank] = -1
                format_count+=1
            elif all_probability[prob_rank] > 1:
                large.append(prob_rank)
            else:
                small.append(prob_rank)

        if format_count==num_probability:
            return prab,alias

        while 1:
            if len(small)==0:
                break
            if len(large)==0:
                break
            small_rank = small.pop()
            small_data = all_probability[small_rank]
            need_data = 1 - small_data
            large_rank = large.pop()
            rest_data = all_probability[large_rank] - need_data

            prab[small_rank] = small_data
            alias[small_rank] = large_rank
            all_probability[large_rank]=rest_data

            if rest_data == 1:
                prab[large_rank] = 1
                alias[large_rank] = -1

            elif rest_data > 1:
                large.append(large_rank)
            else:
                small.append(large_rank)

        while len(small)!=0:
            small_rank=small.pop()
            prab[small_rank]=1
        while len(large)!=0:
            large_rank=large.pop()
            prab[large_rank]=1

        return prab, alias

    def alias_sample(self, prab, alias,rank=None):
        if rank==None:
            rank=int(random.random()*len(prab))
        prab_ = random.random()
        if prab_ < prab[rank]:

            return rank
        else:
            return alias[rank]
    def batch_alias_sample(self,prab,alias,rank_list):
        all_index=[]
        for rank in rank_list:
            index=self.alias_sample(prab,alias,rank)
            all_index.append(index)
        return all_index

    # kon
    def gen_prob_dist(self,N):
        p = np.random.randint(0, 100, N)
        return p / np.sum(p)

    def simulate(self,N=100, k=10000):

        truth = self.gen_prob_dist(N)

        area_ratio = truth * N
        prab, alias = self.generate_alias_table(all_probability=area_ratio)

        ans = np.zeros(N)
        for _ in range(k):
            i = self.alias_sample(alias=alias,prab=prab,rank=_)

            ans[i] += 1


        return ans / np.sum(ans), truth

if __name__=='__main__':
    tool=optimize_funcation()
    tool.simulate()