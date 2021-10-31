# -*- coding: <UTF-8> -*-
import copy
import time
import math
import numpy as np
from Func import *

def get_key_words(passage, key_words_n = 20):
    tfidf_sum = {word:get_TF(passage, word) * get_IDF(passages, word) for word in passage}
    return dict(sorted(tfidf_sum.items(), key=lambda x:x[1], reverse=True)[:key_words_n])

if __name__ == '__main__':
    start_time = time.time()
    print(f'start time is {start_time}')

    with open("199801_clear.txt", "r", encoding='gbk') as f:
        txt = f.readlines()

    passages = []
    d = {}
 
    for line in txt:
        #split paragraphs by '\n'
        if line == '\n':
            if len(d) > 0:
                passages.append(copy.deepcopy(d))
            d = {}
            continue
        
        #split words by spaces & calculating words frequency
        line_list = line.split('  ')
        for term in line_list:
            if term[-2:] in ['/w', '/m', '\n', '/u', '/r', '/c', '/p', '/q', '/d']:
                continue
            d[term] = d.get(term, 0)+1

    passages.append(copy.deepcopy(d))

    passages_keyword = [get_key_words(p, 20) for p in passages]

    key_time = time.time()
    print(f'key time is {key_time - start_time}')
    
    l = len(passages_keyword)
    similarity_matrix = np.zeros([l, l])
    for i in range(l):
        for j in range(i+1, l):
            similarity_matrix[i][j] = get_cosine_similarity(passages_keyword[i], passages_keyword[j])

    end_time = time.time()
    print(f'end time is {end_time}, time spent is {end_time-start_time}')
