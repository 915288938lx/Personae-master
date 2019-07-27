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
    def __init__(self, session, env, seq_length, x_space, y_space, **options): #algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        # session 的初始化时通过父类BaseSLTFModel的初始化完成的,而真正的初始化又是通过BaseSLTFModel的父类BaseTFModel的初始化完成的,BaseTFModel是顶层对象, 接收object参数
        # session 的初始化最终通过Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count
        super(Algorithm, self).__init__(session, env, **options)


        self.seq_length, self.x_space, self.y_space = seq_length, x_space, y_space # x_space = env.data_dim = 4*5 = 20, y_space = env.code_count = len(self.codes) = 4

        try:
            self.hidden_size = options['hidden_size'] #有往**options中传入hidden_size == 5
        except KeyError:
            self.hidden_size = 1

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.seq_length, self.x_space]) # Tensor("Placeholder:0", shape=(?, 5, 20), dtype=float32),输入数据实际形状是:(len(dates),5,3*特征dim), 三维数组; 用了batch处理后, 那形状就是(batch_size,5,3*特征dim)
        self.label = tf.placeholder(tf.float32, [None, self.y_space]) # Tensor("Placeholder_1:0", shape=(?, 4), dtype=float32),实际标签输出形状是: (len(dates),3), 二维数组,用了batch处理后, 那形状就是(batch_size,5,3*特征dim)

    def _init_nn(self):
        with tf.variable_scope('nn'):
            self.rnn = self.add_rnn(1, self.hidden_size) #
            self.rnn_output, _ = tf.nn.dynamic_rnn(self.rnn, self.x, dtype=tf.float32) #返回形状(?,5,5), dynamic_rnn返回[output,(state.cell_state, state.hidden_state)] ,形状分别为(step_num,batch_size,output_uints)
            self.rnn_output = self.rnn_output[:, -1] # Tensor("nn/strided_slice:0", shape=(?, 5), dtype=float32), 返回(?, 5), 输出最后一个输出, 此时rnn_output形状为(batch_size,output_units)
            self.rnn_output_dense = self.add_fc(self.rnn_output, 16) # Tensor("nn/dense/BiasAdd:0", shape=(?, 16), dtype=float32), 进行一次仿射变换
            self.y = self.add_fc(self.rnn_output_dense, self.y_space) #  Tensor("nn/dense_1/BiasAdd:0", shape=(?, 4), dtype=float32), 再进行一次仿射变换,将其最后映射到y_space空间

    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.label) #Tensor("loss/mean_squared_error/value:0", shape=(), dtype=float32)
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self):
        for step in range(self.train_steps): # 一共需要随机抽取train_steps== 30000 次
            batch_x, batch_y = self.env.get_batch_data(self.batch_size) # <class 'tuple'>: (32, 5, 20),<class 'tuple'>: (32, 4), 每次从seq_data_x中随机抽取32个样本 "某次抽取出batch_x的形状是(32,5,25),batch_y的形状是(32,5)  因为seq_data_x返回形状:(len(dates),5,3*特征dim),三维数组";
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.x: batch_x, self.label: batch_y})#loss ==0.1348294
            if (step + 1) % 1000 == 0:
                logging.warning("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)

    def predict(self, x):
        # 进入predict阶段, 就不需要损失函数的参与了. 只需要到输出层即可,即到self.y
        return self.session.run(self.y, feed_dict={self.x: x})


def main(args):

    # mode = args.mode
    mode = 'train'
    # codes = ["600036"]
    # codes = ["601398", "000651", "601998", "000002"]
    codes = ["600036", "601328", "601998", "601398"]
    # codes = args.codes
    # codes = ["AU88", "RB88", "CU88", "AL88"]
    market = args.market # 调用args_parser.py中的model_launcher_parser对象中的参数“stock”，指定为股票市场
    # market = 'future'
    # train_steps = args.train_steps
    train_steps = 20000 # 训练次数
    training_data_ratio = 0.8
    # training_data_ratio = args.training_data_ratio
    # 初始化股票市场
    env = Market(codes, start_date="2008-01-01", end_date="2019-07-19",  # 可选参数可接受如下字典参数
         **{
            "market": market,
            "use_sequence": True,
            "scaler": MinMaxScaler,
            "mix_index_state": False,
            "training_data_ratio": training_data_ratio,}
                 )


    model_name = os.path.basename(__file__).split('.')[0]

    algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        "mode": mode,
        "hidden_size": 5,
        "enable_saver": True,
        "train_steps": train_steps,
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
    })

    algorithm.run()
    algorithm.eval_and_plot()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())
