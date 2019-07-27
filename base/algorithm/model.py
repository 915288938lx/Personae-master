import tensorflow as tf
import numpy as np
import json

from abc import abstractmethod
from helper import data_ploter
from tensorflow.contrib import rnn
from helper.data_logger import generate_algorithm_logger

#base model 定义了最通用的方法如添加卷积层，添加循环层
class BaseTFModel(object):

    def __init__(self, session, env, **options):
        self.session = session
        self.env = env
        self.total_step = 0

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.logger = options['logger']
        except KeyError:
            self.logger = generate_algorithm_logger('model')

        try:
            self.enable_saver = options["enable_saver"]
        except KeyError:
            self.enable_saver = False

        try:
            self.enable_summary_writer = options['enable_summary_writer']
        except KeyError:
            self.enable_summary_writer = False

        try:
            self.save_path = options["save_path"] # lstm中,在Algorithm的初始化的时候传入了"save_path":os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"
        except KeyError:
            self.save_path = None

        try:
            self.summary_path = options["summary_path"]
        except KeyError:
            self.summary_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    def restore(self):
        self.saver.restore(self.session, self.save_path)

    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()

    def _init_summary_writer(self):
        if self.enable_summary_writer:
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.session.graph)

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        return None, None, None

    @abstractmethod
    def run(self):
        pass
 #  静态方法，添加一个循环神经网络（这里都是LSTM神经元）
    @staticmethod
    def add_rnn(layer_count, hidden_size, cell=rnn.BasicLSTMCell, activation=tf.tanh):
        # hidden_size = 5，神经元序列
        cells = [cell(hidden_size, activation=activation) for _ in range(layer_count)]
        return rnn.MultiRNNCell(cells)
 # 静态方法， 添加一个卷积层和池化层， 返回池化层
    @staticmethod
    def add_cnn(x_input, filters, kernel_size, pooling_size):  #Tensor("Placeholder_1:0", shape=(?, 5, 20, 1), dtype=float32)
        convoluted_tensor = tf.layers.conv2d(x_input, filters, kernel_size, padding='SAME', activation=tf.nn.relu)  #Tensor("conv2d/Relu:0", shape=(?, 5, 20, 2), dtype=float32)
        tensor_after_pooling = tf.layers.max_pooling2d(convoluted_tensor, pooling_size, strides=[1, 1], padding='SAME') #Tensor("max_pooling2d/MaxPool:0", shape=(?, 5, 20, 2), dtype=float32)
        return tensor_after_pooling
# 静态方法， 添加一个dense全连接层
    @staticmethod
    def add_fc(x, units, activation=None):
        return tf.layers.dense(x, units, activation=activation)

# 第二级model，base RL增强学习， 传入第一级的baseTFModel类
class BaseRLTFModel(BaseTFModel):
    #本类初始化
    def __init__(self, session, env, a_space, s_space, **options):
        # 父类初始化
        super(BaseRLTFModel, self).__init__(session, env, **options)

        # Initialize evn parameters.
        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.epsilon = options['epsilon']
        except KeyError:
            self.epsilon = 0.9

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 10000

        try:
            self.save_episode = options["save_episode"]
        except KeyError:
            self.save_episode = 10
    # 取值
    def eval(self):
        self.mode = 'test'
        s = self.env.reset('eval')
        while True:
            # 前向传播
            c, a, _ = self.predict(s)  # s为输入数据, c, a, _
            # 前向传播
            s_next, r, status, info = self.env.forward(c, a)
            s = s_next
            if status == self.env.Done:
                self.env.trader.log_asset(0)
                break

    def plot(self):
        with open(self.save_path + '_history_profits.json', mode='w') as fp:
            json.dump(self.env.trader.history_profits, fp, indent=True)

        with open(self.save_path + '_baseline_profits.json', mode='w') as fp:
            json.dump(self.env.trader.history_baselines, fp, indent=True)

        data_ploter.plot_profits_series(
            self.env.trader.history_baselines,
            self.env.trader.history_profits,
            self.save_path
        )
    # 定义checkpoint保存模型参数
    def save(self, episode):
        self.saver.save(self.session, self.save_path)
        self.logger.warning("Episode: {} | Saver reach checkpoint.".format(episode))
    # 保存交易
    @abstractmethod
    def save_transition(self, s, a, r, s_next):
        pass
    # 抽象方法，log损失
    @abstractmethod
    def log_loss(self, episode):
        pass
    # 获取切片片段,如果a大于三分之一，取2，否则如果a小于负三分之一，取3，否则取0，最后结果转换成float32，转换成list
    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 2, np.where(a < - 1 / 3, 1, 0)).astype(np.int32)[0].tolist()
        return a

    def get_stock_code_and_action(self, a, use_greedy=False, use_prob=False):
        # Reshape a.
        if not use_greedy:
            a = a.reshape((-1,))
            # Calculate action index depends on prob.
            if use_prob:
                # Generate indices.
                a_indices = np.arange(a.shape[0])
                # Get action index.
                action_index = np.random.choice(a_indices, p=a)
            else:
                # Get action index.
                action_index = np.argmax(a)
        else:
            if use_prob:
                # Calculate action index
                if np.random.uniform() < self.epsilon:
                    action_index = np.floor(a).astype(int)
                else:
                    action_index = np.random.randint(0, self.a_space)
            else:
                # Calculate action index
                action_index = np.floor(a).astype(int)

        # Get action
        action = action_index % 3
        # Get stock index
        stock_index = np.floor(action_index / 3).astype(np.int)
        # Get stock code.
        stock_code = self.env.codes[stock_index]

        return stock_code, action, action_index

# 并列第二级，监督学习层， 用baseTFModel传入,增加了一些特有的一些全局变量和方法
class BaseSLTFModel(BaseTFModel):

    def __init__(self, session, env, **options):
        super(BaseSLTFModel, self).__init__(session, env, **options)

        # Initialize parameters.
        self.x, self.label, self.y, self.loss = None, None, None, None

        try:
            self.train_steps = options["train_steps"]
        except KeyError:
            self.train_steps = 30000

        try:
            self.save_step = options["save_step"]
        except KeyError:
            self.save_step = 1000
    #如果是训练模式，train，则训练，否则读取训练好的参数
    def run(self):
        if self.mode == 'train':
            self.train()
        else:
            self.restore()

    def save(self, step):
        self.saver.save(self.session, self.save_path)
        self.logger.warning("Step: {} | Saver reach checkpoint.".format(step + 1))

    def eval_and_plot(self):

        x, label = self.env.get_test_data()  # x的形状为(561,5,25), label的形状为(561,5) ; 分别返回整个测试集的所有输入序列 和 整个测试集的输出序列,对应着每个输入序列后一天的close

        y = self.predict(x) #形状为(561,5) , 0.2*总数据=561

        with open(self.save_path + '_y.json', mode='w') as fp:
            json.dump(y.tolist(), fp, indent=True) #保存为model_y.json 在checkpoints\SL\stock\model_y.json 保存着预测值y

        with open(self.save_path + '_label.json', mode='w') as fp:
            json.dump(label.tolist(), fp, indent=True) # #保存为model_label.json 在checkpoints\SL\stock\model_label.json 保存着测试集的真实的标签值

        data_ploter.plot_stock_series(self.env.codes, # env.codes == <class 'list'>: ['600036', '601328', '601998', '601398']
                                      y, # 形状为(561,5)
                                      label, # label的形状为(561,5)
                                      self.save_path) # 保存着和单独运行plot_prices.py一样的结果的图片


class BasePTModel(object):

    def __init__(self, env, **options):

        self.env = env

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        pass

    @abstractmethod
    def restore(self):
        pass

    @abstractmethod
    def run(self):
        pass


class BaseRLPTModel(BasePTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(BaseRLPTModel, self).__init__(env, **options)

        self.env = env

        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def save_transition(self, s, a, r, s_n):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 2, np.where(a < - 1 / 3, 1, 0)).astype(np.int32)[0].tolist()
        return a
