#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np


class Individual(object):

    def __init__(self, net_info):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        self.mutation(1.0)

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            if n >= self.net_info.node_num:    # output node
                in_num = self.net_info.out_in_num[t]
            else:    # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i+1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i+1])

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def __force_mutate(self, current, max_int):
        return (current + np.random.randint(max_int-1) + 1) % max_int

    def mutation(self, mutation_rate=0.01):
        active_check = False

        # intermediate node
        for n in range(self.net_info.node_num):
            t = self.gene[n][0]
            # mutation for type gene
            if np.random.rand() < mutation_rate and self.net_info.func_type_num > 1:
                self.gene[n][0] = self.__force_mutate(self.gene[n][0], self.net_info.func_type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = int(n / self.net_info.rows)
            connect_num = col * self.net_info.rows + self.net_info.input_num
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and connect_num > 1:
                    self.gene[n][i+1] = self.__force_mutate(self.gene[n][i+1], connect_num)
                    if self.is_active[n] and i < self.net_info.func_in_num[t]:
                        active_check = True

        # output node
        for n in range(self.net_info.node_num, self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene node
            if np.random.rand() < mutation_rate and self.net_info.out_type_num > 1:
                self.gene[n][0] = self.__force_mutate(self.gene[n][0], self.net_info.out_type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            connect_num = self.net_info.node_num + self.net_info.input_num
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and connect_num > 1:
                    self.gene[n][i+1] = self.__force_mutate(self.gene[n][i+1], connect_num)
                    if self.is_active[n] and i < self.net_info.out_in_num[t]:
                        active_check = True

        self.check_active()
        return active_check

    def silent_mutation(self, mutation_rate=0.01):
        # intermediate node
        for n in range(self.net_info.node_num):
            t = self.gene[n][0]
            # mutation for type gene
            if not self.is_active[n] and np.random.rand() < mutation_rate and self.net_info.func_type_num > 1:
                self.gene[n][0] = self.__force_mutate(self.gene[n][0], self.net_info.func_type_num)
            # mutation for connection gene
            col = int(n / self.net_info.rows)
            connect_num = col * self.net_info.rows + self.net_info.input_num
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= self.net_info.func_in_num[t]) and np.random.rand() < mutation_rate \
                        and connect_num > 1:
                    self.gene[n][i+1] = self.__force_mutate(self.gene[n][i+1], connect_num)

        # output node
        for n in range(self.net_info.node_num, self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene node
            if not self.is_active[n] and np.random.rand() < mutation_rate and self.net_info.out_type_num > 1:
                self.gene[n][0] = self.__force_mutate(self.gene[n][0], self.net_info.out_type_num)
            # mutation for connection gene
            connect_num = self.net_info.node_num + self.net_info.input_num
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= self.net_info.out_in_num[t]) and np.random.rand() < mutation_rate \
                        and connect_num > 1:
                    self.gene[n][i+1] = self.__force_mutate(self.gene[n][i+1], connect_num)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval

    def active_net_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i+1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


# (1 + \lambda)-ES
class CGP(object):

    def __init__(self, net_info, eval_func, lam=4):
        self.lam = lam
        self.pop = [Individual(net_info) for _ in range(1 + self.lam)]
        self.eval_func = eval_func

        self.num_gen = 0
        self.num_eval = 0

    def _evaluation(self, pop, eval_flag):
        # create network list
        net_lists = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            net_lists.append(pop[i].active_net_list())

        # evaluation
        fp = self.eval_func(net_lists)
        for i, j in enumerate(active_index):
            pop[j].eval = fp[i]
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval

        self.num_eval += len(net_lists)
        return evaluations

    def _log_data(self, net_info_type='active_only'):
        log_list = [self.num_gen, self.num_eval, time.clock(), self.pop[0].eval, self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape((net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    def evolution(self, max_eval=100, mutation_rate=0.01, log_file='./log.txt'):
        with open(log_file, 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            eval_flag = np.empty(self.lam)

            self._evaluation([self.pop[0]], np.array([True]))
            print(self._log_data(net_info_type='active_only'))

            while self.num_eval < max_eval:
                self.num_gen += 1

                # reproduction
                for i in range(self.lam):
                    self.pop[i+1].copy(self.pop[0])    # copy a parent
                    eval_flag[i] = self.pop[i+1].mutation(mutation_rate)    # mutation

                # evaluation and selection
                evaluations = self._evaluation(self.pop[1:], eval_flag=eval_flag)
                best_arg = evaluations.argmax()
                if evaluations[best_arg] >= self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg+1])

                # display and save log
                if eval_flag.sum() > 0:
                    print(self._log_data(net_info_type='active_only'))
                    writer.writerow(self._log_data(net_info_type='full'))


# 修正版CGP: active nodeに変更がある個体をGPUの個数分作る，active nodeに変更がない個体を1個体作る
class ModifyCGP(CGP):

    def __init__(self, net_info, eval_func, gpu_num):
        self.gpu_num = gpu_num
        super(ModifyCGP, self).__init__(net_info, eval_func, lam=self.gpu_num)

    def evolution(self, max_eval=100, mutation_rate=0.01, log_file='./log.txt'):
        with open(log_file, 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            eval_flag = np.empty(self.gpu_num)

            active_num = self.pop[0].count_active_node()
            while active_num < self.pop[0].net_info.min_active_num or active_num > self.pop[0].net_info.max_active_num:
                self.pop[0].mutation(1.0)
                active_num = self.pop[0].count_active_node()
            self._evaluation([self.pop[0]], np.array([True]))
            print(self._log_data(net_info_type='active_only'))

            while self.num_eval < max_eval:
                self.num_gen += 1

                # reproduction
                for i in range(self.gpu_num):
                    eval_flag[i] = False
                    self.pop[i+1].copy(self.pop[0])  # copy a parent
                    active_num = self.pop[i+1].count_active_node()
                    while not eval_flag[i] or active_num < self.pop[i+1].net_info.min_active_num \
                            or active_num > self.pop[i+1].net_info.max_active_num:
                        self.pop[i+1].copy(self.pop[0])    # copy a parent
                        eval_flag[i] = self.pop[i+1].mutation(mutation_rate)    # mutation
                        active_num = self.pop[i+1].count_active_node()

                # evaluation and selection
                evaluations = self._evaluation(self.pop[1:], eval_flag=eval_flag)
                best_arg = evaluations.argmax()
                if evaluations[best_arg] > self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg+1])
                else:
                    self.pop[0].silent_mutation(mutation_rate)

                # display and save log
                print(self._log_data(net_info_type='active_only'))
                writer.writerow(self._log_data(net_info_type='full'))
