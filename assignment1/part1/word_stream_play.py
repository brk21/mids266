from itertools import combinations
from collections import defaultdict, Counter

words = 'hello world how are you today'.split()
table = []
C = 2
counter = Counter()
for i in xrange(len(words)):
    window = words[i:i+C+1]
    if len(window) > 1:
        combos = combinations(window, C)
        counter += Counter(combos)
for key in sorted(counter.iterkeys()):
    table.append(tuple(list(sorted(key)) + [counter[key]*1.00/C]))
sorted(table)

#############
############

temp = {}
table = []
ddict = defaultdict(int)
for i in xrange(len(words)):
    if i-C >= 0 and i+C < len(words):
        window = words[i-C:i+C+1]
        temp1 = []
        temp2 = []
        for j in xrange(len(window)):
            if j != C:
                W1 = window[C]
                W2 = window[j]
                temp1.append(tuple([W1]+[W2]))
                temp2.append(abs(C-j))
        for i,tup in enumerate(temp1):
            ddict[tup] += 1.00/temp2[i]
            
for key in sorted(ddict.iterkeys()):
    table.append(tuple(list(key) + [ddict[key]]))
                
                

sorted(temp1)


            #prelim(sorted([window[0],window[i+1]])) += 1.00/C
            
            
unigrams = {'hello': 5, 'world': 3}
bigrams = {('hello', 'world'): 8}
bigram = ('hello', 'world')

bigram[1]

unigrams[bigram[1]]



