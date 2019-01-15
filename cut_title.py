#coding:utf-8


import sys
import os
import numpy as np

file_path = sys.argv[1]
save_name = "data/"+file_path.split("/")[-1]

data = []
poem_len = []
uniq_words = []
with open(file_path, 'r') as fr:
    for i, line in enumerate(fr.readlines()):
        text = line.strip()#.split(':')[1]
        if i < 10:
            print(text)
        # data.append(text)
        poem_len.append(len(text))
        uniq_words.extend(list(text))


print("avg length:", np.mean(poem_len))
print("count words:", len(set(uniq_words)))
#
# with open(save_name, 'w') as fw:
#     for i in data:
#         fw.write(i+'\n')