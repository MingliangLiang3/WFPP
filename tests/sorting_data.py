import pandas as pd
from open_clip.tokenizer import SimpleTokenizer
import numpy as np
import json
from tqdm import tqdm

df_cc12m = pd.read_csv('../data/cc12m/cc12m_train.csv', sep='\t')
tokenizer = SimpleTokenizer()

with open('../data/cc12m/cc12m_word_frequency_1e7.json', 'r', encoding='utf-8') as f:
    word_frequency_dict = json.load(f)

texts_sw_list = []
for index, row in tqdm(df_cc12m.iterrows(), total=df_cc12m.shape[0]):
    # text = tokenizer.encode(row['caption']) # encode text to tokens
    text = tokenizer.encode_text(row['caption']) # encode text to words
    text = text[:30]
    word_frequency = [word_frequency_dict.get(str(word), 1) for word in text]
    word_frequency = [1 if fq < 0 else fq for fq in word_frequency]

    # text without normalization
    # text_list.append((1 - np.prod(word_frequency)))

    # \log(s(t_i)) = \frac{1}{n} \sum_{i=1}^n \log P(w_i)
    # normalize text by length of text
    texts_sw_list.append((1 - np.prod(word_frequency)) / len(word_frequency))

df_cc12m['texts_sw'] = texts_sw_list
df_cc12m.sort_values(by='texts_sw', ascending=False, inplace=True)
df_cc12m.to_csv('../data/cc12m/cc12m_train_normal.csv', sep='\t', index=False)
# sample first 50% of the data
df_cc12m.head(int(df_cc12m.shape[0] / 2)).to_csv('../data/cc12m/cc12m_train_normal_sample.csv', sep='\t', index=False)
