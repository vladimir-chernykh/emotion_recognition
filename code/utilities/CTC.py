# coding:utf-8
# CTC utility functions
# Author    :  David Leon (Dawei Leng)
# Created   :   9, 28, 2015
# Revised   :   7,  5, 2016
# All rights reserved
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import numpy as np
from numba import jit

class CTC(object):
    @classmethod
    def best_path_decode_batch(self, scorematrix, scorematrix_mask=None, blank_symbol=None):
        """
        Computes the best path by simply choosing most likely label at each time step, batch version
        :param scorematrix: (B, T, C+1)
        :param scorematrix_mask: (B, T)
        :param blank_symbol: = C by default
        :return: resultseqs <list of size = B>
        """
        B, T, C1 = scorematrix.shape
        if blank_symbol is None:
            blank_symbol = C1 - 1
        resultseqs = []
        for i in range(B):
            if scorematrix_mask is None:
                t = T
            else:
                submask = scorematrix_mask[i, :]
                t = int(submask.sum())
            submatrix = scorematrix[i,:t, :]
            result = self.best_path_decode(submatrix.T, blank_symbol)
            resultseqs.append(result)
        return resultseqs

    @staticmethod
    def best_path_decode(scorematrix, blank_symbol=-1):
        """
        Computes the best path by simply choosing most likely label at each time step
        :param scorematrix: (C+1, T)
        :param blank_symbol: position of blank label, either = -1 or 0
        :return:
        """
        # Compute best path
        best_path = np.argmax(scorematrix, axis=0).tolist()
        C1, T = scorematrix.shape
        blank_symbol = (C1 + blank_symbol) % C1
        # print('blank_symbol = ', blank_symbol)

        result = []
        for i, b in enumerate(best_path):
            # ignore blanks
            if b == blank_symbol:
                continue
            # ignore repeats
            elif i != 0 and b == best_path[i - 1]:
                continue
            else:
                result.append(b)
        return result

    @classmethod
    def calc_CER(self, resultseqs, targetseqs):
        assert(len(resultseqs) == len(targetseqs))
        total_len, total_edit_dis = 0.0, 0.0
        for resultseq, targetseq in zip(resultseqs, targetseqs):
            total_len += len(targetseq)
            if resultseq is not None and len(resultseq) > 0:
                edit_dis = self.editdist(resultseq, targetseq)
            else:
                edit_dis = len(targetseq)
            total_edit_dis += edit_dis
        CER = total_edit_dis / total_len * 100.0
        return CER, total_edit_dis, total_len

    @staticmethod
    @jit(nopython=True, cache=True)
    def editdist(s, t):
        """
        From Wikipedia article; Iterative with two matrix rows.
        """
        if s == t:
            return 0
        elif len(s) == 0:
            return len(t)
        elif len(t) == 0:
            return len(s)
        v0 = [0] * (len(t) + 1)
        v1 = [0] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
        return v1[len(t)]

if __name__ == '__main__':
    d = CTC.editdist([1,2,3],[2,3])
    print('d = ', d)