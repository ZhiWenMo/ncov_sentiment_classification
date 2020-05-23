import numpy as np
import pandas as pd
from utils import compute_inputs, compute_outputs, f1_macro
import tensorflow as tf
import tensorflow.keras.backend as K
from importlib import import_module
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description='nCov sentiment Classification')
parser.add_argument('--model', default='BERT_base', type=str, help='choose a model: BERT_base, ROBERTA_large')
parser.add_argument('--use_multi_gpu', default=False, type=bool, help='if True use multiple gpu for training')
parser.add_argument('--use_pl', default=True, type=bool, help='when True you should set the path of pseudo label')
args = parser.parse_args()

model_name = args.model
# import the config with model
x = import_module('models.'+model_name)
config = x.Config(pl=args.use_pl)
print(config.bert_path)

# handel the data for model inputs
df_train = pd.read_csv(config.train_data_path, engine='python', encoding='utf-8')
df_train = df_train[df_train[config.output_categories].isin(config.labels)]
if config.n_sample:
    df_train = df_train.sample(config.n_sample)

df_test = pd.read_csv(config.test_data_path, engine='python', encoding='utf-8')

inputs, outputs = compute_inputs(df_train, config), compute_outputs(df_train, config)
test_inputs = compute_inputs(df_test, config)

# generate kfold data for pseudo labeling
pl_train_idxs = []
if config.pl_data_path:
    df_pl = pd.read_csv(config.pl_data_path, engine='python', encoding='utf-8')
    pl_inputs, pl_outputs = compute_inputs(df_pl, config), compute_outputs(df_pl, config)
    pl_gkf = StratifiedKFold(n_splits=config.num_of_fold).split(X=df_pl[config.input_categories].fillna('-1'),
                                                                y=df_pl[config.output_categories].fillna('-1'))
    for fold, (train_id, valid_id) in enumerate(pl_gkf):
        pl_train_idxs.append(train_id)


skf = StratifiedKFold(n_splits=config.num_of_fold).split(X=df_train[config.input_categories].fillna('-1'),
                                        y=df_train[config.output_categories].fillna('-1'))

valid_oof = np.zeros_like(outputs)
test_oof = []

for fold, (train_idx, valid_idx) in enumerate(skf):
    if not pl_train_idxs: # if don't use pseudo labeling
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]
    else: # make sure psedudo labeling data in training set, concat and shuffle
        pl_train_idx = pl_train_idxs[fold]
        train_inputs = [np.concatenate([inputs[i][train_idx], pl_inputs[i][pl_train_idx]]) for i in range(len(inputs))]
        train_outputs = np.concatenate([outputs[train_idx], pl_outputs[pl_train_idx]])

        shuffled_data = shuffle(*train_inputs, train_outputs)

        train_inputs = shuffled_data[:-1]
        train_outputs = shuffled_data[-1]

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = outputs[valid_idx]

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    K.clear_session()

    if bool(args.use_multi_gpu): # use multiple gpu
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = x.SentimentClfModel(config)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_macro])
    else: # use single gpu
        model = x.SentimentClfModel(config)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_macro])

    model.fit(train_inputs, train_outputs,
              validation_data=[valid_inputs, valid_outputs],
              epochs=config.num_epochs, batch_size=config.train_batch_size)

    model.save_weights(f'{config.model_name}_{fold}.h5')
    valid_oof[valid_idx] = model.predict(valid_inputs, config.test_batch_size)
    test_oof.append(model.predict(test_inputs, config.test_batch_size))

sub = np.average(test_oof, axis=0)
np.save(f'{config.model_name}_pred_proba.npy', sub)

np.save(f'{config.model_name}_valid_oof.npy', valid_oof)
np.save(f'{config.model_name}_outputs.npy', outputs)
