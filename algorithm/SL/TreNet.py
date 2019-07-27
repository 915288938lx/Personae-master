# coding=utf-8

import tensorflow as tf
import logging
import os

from algorithm import config
from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from base.algorithm.model import BaseSLTFModel
from sklearn.preprocessing import MinMaxScaler
from helper.args_parser import model_launcher_parser


class Algorithm(BaseSLTFModel):
    def __init__(self, session, env, seq_length, x_space, y_space, **options):
        super(Algorithm, self).__init__(session, env, **options)
        #
        #
        # 形状:(len(dates),5,3*特征dim),三维数组
        self.seq_length, self.x_space, self.y_space = seq_length, x_space, y_space #   x_space = env.data_dim, y_space = env.code_count

        try:
            self.hidden_size = options['hidden_size']
        except KeyError:
            self.hidden_size = 1
        # 下方所有方法在 class Algorithm 初始化的时候已经实例化
        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()


    # 构建类的时候已经初始化了
    def _init_input(self):
        self.rnn_x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space]) # 形状为(?,5,20) x_space = env.data_dim, y_space = env.code_count
        self.cnn_x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space, 1]) # 形状为(?, 5,20,1)   x_space = env.data_dim, y_space = env.code_count. 这里的1是指的是输入通道数, 股票数据为1个通道
        self.label = tf.placeholder(tf.float32, [None, self.y_space]) # 形状为(?, 4)


    # 构建类的时候已经初始化了
    def _init_nn(self):
        # self.add_rnn由传入的参数BaseSLTFModel的父类BaseTFModel下的静态方法 @staticmethod def add_rnn（）所初始化，返回cells，这里是BasioLSTMCell序列
        #    def add_rnn(layer_count, hidden_size, cell=rnn.BasicLSTMCell, activation=tf.tanh):
        #       return rnn.MultiRNNCell(cells)
        self.rnn = self.add_rnn(1, self.hidden_size) # 返回LSTMCell ,  "hidden_size": 5
        self.rnn_output, c_t_and_h_t = tf.nn.dynamic_rnn(self.rnn, self.rnn_x, dtype=tf.float32) # rnn_output的形状为(?,5,5)
        self.rnn_output = self.rnn_output[:, -1] #rnn_output形状变为(?,5)
        # self.cnn_x_input is a [-1, 5, 20, 1] tensor, after cnn, the shape will be [-1, 5, 20, 5]. ???????????????????????????应该是[-1, 5, 20, 2]吧
        # 为什么是20? 因为作者在做这个项目的时候用了4只股票, 每只股票5个特征值['open', 'high', 'low', 'close', 'volume'],
        # 同时place_holder 中是这样定义的(None,self.length,self.x_space,1), 也就是说self.x_space = env.data_dim = 20
        # 那是因为cnn的输入形状就是这样的!  tensor: (batch_size, width, height, channels)
        # add_cnn 返回一个卷积层叠加最大池化层，最终返回最大池化层
        self.cnn = self.add_cnn(self.cnn_x, filters=2, kernel_size=[2, 2], pooling_size=[2, 2]) #这里返回的是一个张量,形状为(?,5,20,2)
        self.cnn_output = tf.reshape(self.cnn, [-1, self.seq_length * self.x_space * 2]) #  展平除了批量维度以外的所有其他维度, 返回的形状为(?,200) 5*20*2 == 200 ,keras中的实现是model.add(layers.Flatten()),"首先，
        # 我们需要将3D 输出展平为1D",注意这里是没有用激活函数的,所以用矩阵变形就可以达到目的
        # 特征融合层
        self.y_concat = tf.concat([self.rnn_output, self.cnn_output], axis=1) #矩阵拼接,特征融合层,返回的形状为(?, 205)
        # 添加一个dense全连接层,full connected layer
        self.y_dense = self.add_fc(self.y_concat, 16) # 添加全连接层,返回形状为 (?,16) #
        # 添加一个dense全连接层

        # ######最终获取到self.y
        self.y = self.add_fc(self.y_dense, self.y_space) #返回形状为(? , 4)  改变y_dense 的最后一维度为y_space


    # 构建类的时候已经初始化了
    # 传入最终输出，其实就是loss_function
    def _init_op(self):
        # Algrithm在初始化的时候已经初始化这个方法，继而两个tf.variable_scope也进入内存变量

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.label)


        #
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) # learning rate = 0.001
            # 在tf.variable_scope('train')数据域下， 执行op，即最小化loss值
            self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())
    # 开始训练#  train_steps,default=100000
    def train(self):
        for step in range(self.train_steps):#  train_steps,default=100000

            # batch_x,batch_y分别是第n天到第n+32天、第n+1天到第n+1+32 天的数据；其实，batch_y 也是label标签
            batch_x, batch_y = self.env.get_batch_data(self.batch_size) #<class 'tuple'>: (32, 5, 20)
            # 将batch_x 数据分别赋值给循环网络、卷积网络，注意和def ——init_input() 方法中数据的形状相对应
            x_rnn, x_cnn = batch_x, batch_x.reshape((-1, self.seq_length, self.x_space, 1))#(32，5，20)，(32，5，20，1)
            # run session 其中包括执行train_op，两步op，一是让loss实际执行，二十让train_op实际执行
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.rnn_x: x_rnn,  #注意和def ——init_input() 方法中数据的形状相对应
                                                                              self.cnn_x: x_cnn,  #注意和def ——init_input() 方法中数据的形状相对应
                                                                              self.label: batch_y})#注意和def ——init_input() 方法中数据的形状相对应
            if (step + 1) % 1000 == 0:
                logging.warning("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)
    # 进入到predict阶段,已经不需要损失函数的参与了
    def predict(self, x):
        return self.session.run(self.y, feed_dict={self.rnn_x: x,
                                                   self.cnn_x: x.reshape(-1, self.seq_length, self.x_space, 1)})


def main(args):

    # mode = args.mode
    mode = "train"
    # codes = args.codes
    # codes = ["601398"]
    codes = ["600036", "601328", "601998", "601398"]

    # codes = ["AU88", "RB88", "CU88", "AL88"]
    market = args.market # default="stock"
    train_steps = args.train_steps #  default=100000
    training_data_ratio = 0.8
    # training_data_ratio = args.training_data_ratio


    # env为股票市场market， market市场实例化，**可选参数传入
    env = Market(codes, start_date="2008-01-01", end_date="2019-07-19", **{
        "market": market,## default="stock"
        "use_sequence": True,
        "scaler": MinMaxScaler, # sklearn提供的缩放器
        "mix_index_state": False,# 表明要混合上证指数,以结合市场趋势做更宏观的预测
        "training_data_ratio": training_data_ratio,
    })
    model_name = os.path.basename(__file__).split('.')[0] # 返回“TreNet，即文件名
    # 算法初始化，这里是TreNet实例化， 传入一系列**可选参数
    algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        "mode": mode,# test
        "hidden_size": 5, # 应该是5层hidden layer
        "enable_saver": True,
        "train_steps": train_steps,#  default=100000
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
    })

    algorithm.run()
    algorithm.eval_and_plot()


if __name__ == '__main__':
    # 传入包含参数的类 model_launcher_parser.parse_args()
    main(model_launcher_parser.parse_args())
