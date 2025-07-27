import paddle
import numpy as np
from paddle.nn import Embedding, Linear

# 假设vocab字典已加载，需根据实际情况加载vocab
# 这里给出一个简单的加载方式
import os
vocab_path = os.path.join('CED_Dataset', 'dict.txt')
vocab = {}
with open(vocab_path, encoding='utf-8') as f:
    for idx, line in enumerate(f):
        word = line.strip().split()[0]
        vocab[word] = idx
vocab['<pad>'] = len(vocab)

class LstmModel(paddle.nn.Layer):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.dict_dim = vocab['<pad>']
        self.emb_dim = 128
        self.hid_dim = 128
        self.class_dim = 2
        self.embedding = Embedding(self.dict_dim + 1, self.emb_dim, sparse=False)
        self.fc1 = Linear(self.emb_dim, self.hid_dim)
        self.lstm = paddle.nn.LSTM(self.hid_dim, self.hid_dim)
        self.fc2 = Linear(19200, self.class_dim)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        fc_1 = self.fc1(emb)
        x, _ = self.lstm(fc_1)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc2(x)
        x = paddle.nn.functional.softmax(x)
        return x

def load_lstm_model(pdparams_path='LSTM.pdparams'):
    model = LstmModel()
    model_state_dict = paddle.load(pdparams_path)
    model.set_state_dict(model_state_dict)
    model.eval()
    return model

# 示例评估函数（需传入eval_loader）
def evaluate_lstm(model, eval_loader):
    accuracies = []
    losses = []
    for batch_id, data in enumerate(eval_loader):
        sent = data[0]
        label = data[1]
        logits = model(sent)
        loss = paddle.nn.functional.cross_entropy(logits, label)
        acc = paddle.metric.accuracy(logits, label)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())
    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    return avg_acc, avg_loss
