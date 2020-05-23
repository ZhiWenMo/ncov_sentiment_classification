import pandas as pd
import numpy as np


def submit(proba, df_sub):
    sub = np.argmax(proba, axis=1)
    df_sub['y'] = sub - 1
    df_sub.to_csv('submission.csv', index=False, encoding='utf-8')


if __name__ == "__main__":
    df_sub = pd.read_csv('./input0/submit_example.csv', engine='python', encoding='utf-8')
    
    bert_proba = np.load('./temp/BERT_base_pred_proba.npy')
    roberta_proba = np.load('./temp/ROBERTA_large_pred_proba.npy')
    
    sub = np.mean([bert_proba, roberta_proba], axis=0)
    sub = np.argmax(sub, axis=1)
    df_sub['y'] = sub - 1
    df_sub.to_csv('submission.csv', index=False, encoding='utf-8')