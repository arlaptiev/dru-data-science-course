"""
This is an excersise from Yandex course 'Математика и Python для анализа данных'

Дан набор предложений, скопированных с Википедии. Каждое из них имеет
"кошачью тему" в одном из трех смыслов:
- кошки (животные)
- UNIX-утилита cat для вывода содержимого файлов
- версии операционной системы OS X, названные в честь семейства кошачьих
Ваша задача — найти два предложения, которые ближе всего по смыслу к
расположенному в самой первой строке. В качестве меры близости по смыслу мы
будем использовать косинусное расстояни
"""

import re
import numpy as np
import scipy.spatial


def match(l, n=1, m=0):
    """Matches the model sentence to the closest in meaning sentences from
    ----------
    l: list of sentences to match
    m: model sentence index
    n: number of closest sentences to find
    """

    def tokenize(s):
        """Splits all sentences in a list into separate words and lowercases them"""
        arr = []
        for i in range(len(s)):
            arr.append([x for x in re.split('[^a-z]', s[i].lower()) if x])
        return arr

    def all_tokens(s):
        """Returns a dictionary of all unique tokens in a list"""
        tokens = {}
        i = 0
        for r in s:
            for e in r:
                if not e in tokens:
                    tokens[e] = i
                    i += 1
        return tokens

    def count_tokens(s, tokens):
        """Returns a list with counted numbers of tokens in every sentence of s"""
        arr = np.zeros(len(s) * len(tokens)).reshape(len(s), len(tokens))
        for i in range(len(s)):
            for el in s[i]:
                arr[i, tokens[el]] += 1
        return arr

    def cosine_distances(s, m):
        """Returns a list of cosine distances from model sentence to corresponding sentence """
        arr = np.zeros(len(s))
        for i in np.append(np.arange(m), np.arange(m + 1, len(s))):
            arr[i] = scipy.spatial.distance.cosine(s[m], s[i])
        return arr

    def min_dis(x, n):
        """Returns indicies of the first n minimum elements in a list"""
        return x.argsort()[1:n+1]

    s = tokenize(l)
    tokens = all_tokens(s)
    s = count_tokens(s, tokens)
    s = cosine_distances(s, m)
    indicies = min_dis(s, n+1)

    return indicies


if __name__ == '__main__':
    f = open('sentences.txt', 'r', encoding='utf8')
    s = f.read().splitlines()
    indicies = match(s)

    print('Original:\n', s[0])
    print('\nFound:')
    for i in indicies:
        print(s[i])
