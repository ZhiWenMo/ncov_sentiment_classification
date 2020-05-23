import tensorflow as tf
import transformers
from tensorflow.keras.layers import Dense, add, Dropout, LayerNormalization
from .TextCNN import TextCNN


class Config:
    def __init__(self, pl=None):
        self.model_name = 'ROBERTA_large'
        input_path_prefix = '/openbayes/input/'
        self.n_sample = None
        self.train_data_path = input_path_prefix + 'input0/nCoV_100k_train.labled.csv'
        self.test_data_path = input_path_prefix + 'input0/nCov_10k_test.csv'
        self.submit_example_path = input_path_prefix + 'input0/submit_example.csv'
        self.input_categories = '微博中文内容'
        self.output_categories = '情感倾向'
        self.labels = ['-1', '0', '1']
        self.pl_data_path = None
        if pl:
            self.pl_data_path = input_path_prefix + 'input0/test_pseudo_labled.csv'

        self.bert_path = input_path_prefix + 'input1/'
        self.temp_data_path = input_path_prefix + 'temp/'
        self.dropout_rate = 0.15
        self.activation = 'elu'

        self.num_of_fold = 5
        self.max_len = 140
        self.num_epochs = 1
        self.train_batch_size = 32
        self.test_batch_size = 16
        self.learning_rate = 2e-5

        self.kernel_size = (8, 16, 32)
        self.filter_size = 256

        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_path + 'vocab.txt')


class SentimentClfModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(SentimentClfModel, self).__init__(**kwargs)
        self.config = config
        self.bert_model_config = transformers.BertConfig. \
            from_pretrained(self.config.bert_path+'config.json')
        self.bert_model = transformers.TFBertModel. \
            from_pretrained(self.config.bert_path+'pytorch_model.bin',
                            from_pt=True,
                            config=self.bert_model_config)

        self.textcnn = TextCNN(kernel_sizes=self.config.kernel_size, filter_size=self.config.filter_size,
                               activation=self.config.activation, dropout_rate=self.config.dropout_rate)

        self.linear = Dense(len(self.config.kernel_size) * self.config.filter_size, activation=self.config.activation)
        self.layer_norm = LayerNormalization(axis=-1, center=True, scale=True)
        self.dropout = Dropout(self.config.dropout_rate)
        self.out = Dense(len(self.config.labels), activation='softmax')


    def call(self, inputs):
        input_ids, input_mask, token_type_ids = inputs
        embedding, _ = self.bert_model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        textcnn_out = self.textcnn(embedding)
        linear_out = self.linear(textcnn_out)
        add_norm = self.layer_norm(add([textcnn_out, linear_out]))
        add_norm = self.dropout(add_norm)
        output = self.out(add_norm)

        return output

    def compute_output_shape(self, input_shape):
        return (None, len(self.config.labels))

