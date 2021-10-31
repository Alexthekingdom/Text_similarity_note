import math

def get_TF(passage, word):
    return passage[word] / sum(passage.values())

def get_IDF(passages, word):
    return len(passages) / (1 + sum([1 for p in passages if word in p]))

def get_dict_dot(d1, d2):
    return sum([d1.get(k, 0) * d2.get(k, 0) for k in d1])

def get_norm(vec):
    return math.sqrt(sum([num**2 for num in vec]))

def get_cosine_similarity(p1, p2):
    return get_dict_dot(p1, p2) / (get_norm(p1.values()) * get_norm(p2.values()))