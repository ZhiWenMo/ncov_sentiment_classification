import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def compute_outputs(df, config):
    return to_categorical(df[config.output_categories].astype(int).values + 1)


def compute_inputs(df, config):
    input_ids, input_mask, token_type_ids = [], [], []
    for string in tqdm(df[config.input_categories], total=len(df)):
        ids, mask, token_type = convert_strings_to_inputs(str(string), config)
        input_ids.append(ids)
        input_mask.append(mask)
        token_type_ids.append(token_type)

    return list(map(lambda x: np.asarray(x, dtype=np.int32), [input_ids, input_mask, token_type_ids]))


def convert_strings_to_inputs(string, config):

    tokenized_outputs = config.tokenizer.encode_plus(string,
                                                     add_special_tokens=True,
                                                     max_length=config.max_len,
                                                     truncation_strategy='longest_first')
    input_ids = tokenized_outputs['input_ids']
    input_mask = [1] * len(input_ids)
    token_type_ids = tokenized_outputs['token_type_ids']
    padding_id = config.tokenizer.pad_token_id
    padding_length = config.max_len - len(input_ids)

    input_ids = input_ids + ([padding_id] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    return input_ids, input_mask, token_type_ids


# metric

def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


if __name__ == '__main__':
    import pandas as pd
    from models.BERT_base import Config
    config = Config()

    df_train = pd.read_csv(config.train_data_path, engine='python', encoding='utf-8')
    df_train = df_train[df_train[config.output_categories].isin(config.labels)]

    test_df = df_train.sample(1000)

    inputs = compute_inputs(test_df, config)
    outputs = compute_outputs(test_df, config)

    print(inputs[0].shape)
    print(outputs.shape)

