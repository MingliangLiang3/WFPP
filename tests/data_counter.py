import pandas as pd
from open_clip.tokenizer import SimpleTokenizer
import collections
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool

reader = pd.read_csv('../data/cc12m/cc12m_train.csv', sep='\t', chunksize=10000)
tokenizer = SimpleTokenizer()

def process_by_chunk(row):
    text = row[1]['caption']
    # return tokenizer.encode(text) # return tokens of text
    return tokenizer.encode_text(text) # return words of text

words = []
with Pool(processes=16) as pool:
    for chunk in tqdm(reader):
        df_list = pool.map(process_by_chunk, chunk.iterrows())
        words.extend(df_list)

counter = collections.Counter([tk for st in words for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
counter = dict(counter.items())
total_count = sum(counter.values())
counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
print("Total words:", total_count)
# save the counter to a json file
with open('../data/cc12m/cc12m_train_counter.json', 'w', encoding='utf-8') as f:
    json.dump(counter, f)
freqs = {word: count / total_count for word, count in counter.items()}  # f(w_i)

threshold = 1e-7
p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in counter}  # P(w_i)
# Save the p_drop to a json file
with open('../data/cc12m/cc12m_word_frequency_1e7.json', 'w', encoding='utf-8') as f:
    json.dump(p_drop, f)
