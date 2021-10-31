# -*- coding: <UTF-8> -*-
import copy
import time
import math
import numpy as np
from Func import *
import multiprocessing
import os

def get_IDF_parallel(word_total, word, passages_n):
    return passages_n / (1 + word_total[word])

def get_key_words_parallel(start, batch_size, word_total, passages_n, passages_batch, key_words_n=20, passages_keyword=[]):
    
    for k in range(batch_size):
        tfidf_sum = {word:get_TF(passages_batch[k], word) * get_IDF_parallel(word_total, word, passages_n) for word in passages_batch[k]}
        passages_keyword[start+k] = copy.deepcopy(dict(sorted(tfidf_sum.items(), key=lambda x:x[1], reverse=True)[:key_words_n]))

def get_cosine_similarity_parallel(start, batch_size, l, passages_keyword, similarity_matrix):
    for k in range(batch_size):
        tmp = []
        for j in range(start+k, l-1):
            tmp.append(get_cosine_similarity(passages_keyword[start+k], passages_keyword[j]))
        similarity_matrix[start+k] = copy.deepcopy(tmp) 


if __name__ == '__main__':
    os.chdir('./SeaData')
    start_time = time.time()
    print(f'start time is {start_time}')
    
    with open("199801_clear.txt", "r", encoding='gbk') as f:
        txt = f.readlines()

    passages = []
    d = {}
    word_total = {}
    for line in txt:
        #split paragraphs by '\n'
        if line == '\n':
            if len(d) > 0:
                passages.append(copy.deepcopy(d))
                for k in d:
                    word_total[k] = word_total.get(k, 0)+1
            d = {}
            continue
        
        #split words by spaces & calculating words frequency
        line_list = line.split('  ')
        for term in line_list:
            if term[-2:] in ['/w', '/m', '\n', '/u', '/r', '/c', '/p', '/q', '/d']:
                continue
            d[term] = d.get(term,0)+1
            
            
    passages.append(copy.deepcopy(d))
    passages = passages
    passages_n = len(passages)

    #get keywords
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cores)
    mgr = multiprocessing.Manager()
    passages_keyword = mgr.list(range(passages_n))

    batch_size = math.ceil(passages_n/cores)
    print('batch',batch_size)
    for i in range(cores):
        start = i*batch_size
        pool.apply_async(func=get_key_words_parallel, args=(start, batch_size, word_total, passages_n, passages[start:start+batch_size], 20, passages_keyword))
    
    pool.close()
    pool.join()
    #print('pass',passages_keyword)
    key_time = time.time()
    print(f'key time is {key_time - start_time}')
    
    # get similarity_matrix
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #similarity_matrix = mgr.list(range(passages_n-1))
    #for i in range(cores):
    #    start = i*batch_size
    #    pool.apply_async(func=get_cosine_similarity_parallel, args=(start, batch_size, passages_n, passages_keyword, similarity_matrix))
    #pool.close()
    #pool.join()
    #print(similarity_matrix[-5:])
    #
    #end_time = time.time()
    #print(f'end time is {end_time}, time spent is {end_time-start_time}')


    l = len(passages_keyword)
    similarity_matrix = np.zeros([l, l])
    for i in range(l):
        for j in range(i+1, l-1):
            similarity_matrix[i][j] = get_cosine_similarity(passages_keyword[i], passages_keyword[j])

    end_time = time.time()
    print(f'end time is {end_time}, time spent is {end_time-start_time}')

    