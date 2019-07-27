# coding=utf-8

import pandas as pd
import numpy as np
import math

from base.env.trader import Trader
from base.model.document import Stock, Future
from sklearn.preprocessing import StandardScaler


class Market(object):

    Running = 0
    Done = -1

    def __init__(self, codes, start_date="2008-01-01", end_date="2019-07-19", **options):

        # Initialize codes.
        # 股票代码例如“[600318,600515]”
        self.codes = codes
        #
        self.index_codes = []
        # self.state_codes = self.codes + self.index_codes,  某次state_codes返回为<class 'list'>: ['600036', '601998', 'sh']
        self.state_codes = []

        # Initialize dates.
        self.dates = [] # 全局变量dates
        self.t_dates = []
        self.e_dates = []

        # Initialize data frames.
        self.origin_frames = dict()
        self.scaled_frames = dict()

        # Initialize scaled  data x, y.
        self.data_x = None  #由_init_series_data返回
        self.data_y = None  #由_init_series_data返回

        # Initialize scaled seq data x, y.
        self.seq_data_x = None  #由_init_sequence_data返回 将列表转换成numpy数组 返回形状:(len(dates),5,3*特征dim),三维数组
        self.seq_data_y = None  #由_init_sequence_data返回 将列表转换成numpy数组 返回形状:(len(dates),3),二维数组

        # Initialize flag date.
        # 下个日期
        self.next_date = None
        self.iter_dates = None
        self.current_date = None

        # 初始化可选参数
        self._init_options(**options)

        # 初始化股票数据
        self._init_data(start_date, end_date)
    # 初始化可选参数
    def _init_options(self, **options):

        try:
            self.m_type = options['market']
        except KeyError:
            self.m_type = 'stock'

        try:
            self.init_cash = options['cash']
        except KeyError:
            self.init_cash = 100000
        # 初始化记录器
        try:
            self.logger = options['logger']
        except KeyError:
            self.logger = None
        # 初始化？
        try:
            self.use_sequence = options['use_sequence']
        except KeyError:
            self.use_sequence = False
        # 初始化是否进行归一化处理，如果输错，则默认归一化处理
        try:
            self.use_normalized = options['use_normalized']
        except KeyError:
            self.use_normalized = True
         #初始化是否混合交易信息
        try:
            self.mix_trader_state = options['mix_trader_state']
        except KeyError:
            self.mix_trader_state = True
        try:
            self.mix_index_state = options['mix_index_state']
        except KeyError:
            self.mix_index_state = False
        # 初始化如果mix_index_state存在，则再index_codes后面增加一个‘sh’
        finally:
            if self.mix_index_state:
                self.index_codes.append('sh')
        # # 初始化每个片段序列的长度，如果输出默认是5（天）,如果输入的长度大于1， 则接受， 否则片段长度强制设置为2
        try:
            self.seq_length = options['seq_length']
        except KeyError:
            self.seq_length = 5
        finally:
            self.seq_length = self.seq_length if self.seq_length > 1 else 2
         #初始化训练数据集的比例，如果输入错误，则默认0.7，默认传入main函数中的0.98
        try:
            self.training_data_ratio = options['training_data_ratio']
        except KeyError:
            self.training_data_ratio = 0.7
        # 初始化特征缩放器，默认标准缩放器
        try:
            scaler = options['scaler']
        except KeyError:
            scaler = StandardScaler
        # 展示代码为codes + index_codes，某次state_codes返回为<class 'list'>: ['600036', '601998', 'sh']
        self.state_codes = self.codes + self.index_codes
        self.scaler = [scaler() for _ in self.state_codes] # 返回一个scaler()对象列表, state_codes里面有几个股票代码就初始化几个scaler对象
        self.trader = Trader(self, cash=self.init_cash)
        self.doc_class = Stock if self.m_type == 'stock' else Future

    def _init_data(self, start_date, end_date):
        self._init_data_frames(start_date, end_date) # 初始化原始数据和被scaled的原始数据
        self._init_env_data() # 初始化seq_data和series_data
        self._init_data_indices() # 初始化训练集和测试集的索引集合和日期集合

    def _validate_codes(self):
        if not self.state_code_count:
            raise ValueError("Codes cannot be empty.")
        for code in self.state_codes:
            if not self.doc_class.exist_in_db(code):
                raise ValueError("Code: {} not exists in database.".format(code))

    def _init_data_frames(self, start_date, end_date):
        # Remove invalid codes first.
        self._validate_codes()
        # Init columns and data set.
        columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
        # Load data.
        for index, code in enumerate(self.state_codes):
            # Load instrument docs by code. doc_class 此时为Stock,调用document里面Stock类
            instrument_docs = self.doc_class.get_k_data(code, start_date, end_date) # 返回某个股票代码从start_date到end_date的数据,按date升序排列
            # Init instrument dicts.
            instrument_dicts = [instrument.to_dic() for instrument in instrument_docs] # 调用document文档里的to_dict() 方法,返回这只股票所有行除了_id列以外的数据的to_dict() 值
            # Split dates.分离出date列
            dates = [instrument[1] for instrument in instrument_dicts] #第0索引是tushare下载下来数据的第一列,日期在第二列
            # Split instruments.# 分离出除了_id列和date列以外的所有数据
            instruments = [instrument[2:] for instrument in instrument_dicts]
            # Update dates set.
            dates_set = dates_set.union(dates) #去重返回两个集合的并集 ,经历循环结束后, 会把其他股票代码的日期也放到这个地方, 所以用union返回去重并集合
            # Build origin and scaled frames.
            scaler = self.scaler[index] # 把scaler()对象中第一个scaler()对象赋值到scaler, 用来为第一条数据进行缩放操作
            scaler.fit(instruments) # 对分离的除了_id列和date列以外的所有数据进行缩放
            instruments_scaled = scaler.transform(instruments)
            origin_frame = pd.DataFrame(data=instruments, index=dates, columns=columns)
            scaled_frame = pd.DataFrame(data=instruments_scaled, index=dates, columns=columns)
            # Build code - frame map.建立股票代码-dataframe的对应关系
            self.origin_frames[code] = origin_frame # origin_frames是以dates为索引, 包含5列分别为['open', 'high', 'low', 'close', 'volume']的Dataframe
            self.scaled_frames[code] = scaled_frame # 用分别用Scaler()缩放器缩放过后的origin_frame, 也包含['open', 'high', 'low', 'close', 'volume']等列
        # Init date iter.
        self.dates = sorted(list(dates_set)) #sorted, 对可迭代对象进行升序排序, 最终传递到market的self.dates变量中
        # Rebuild index.
        for code in self.state_codes:
            origin_frame = self.origin_frames[code]
            scaled_frame = self.scaled_frames[code]
            # 对日期做并集、去重之后,将日期做为新索引, 对origin_frames,scaled_frames进行重新索引
            self.origin_frames[code] = origin_frame.reindex(self.dates, method='bfill') # method = 'bfill' 指的是后向填充(或搬运)值
            self.scaled_frames[code] = scaled_frame.reindex(self.dates, method='bfill')# # method = 'bfill' 指的是后向填充(或搬运)值

    def _init_env_data(self):
        if not self.use_sequence:
            self._init_series_data()
        else:
            self._init_sequence_data()

    def _init_series_data(self): #初始化序列数据,
        # Calculate data count.data_count=  总数据行数-1
        self.data_count = len(self.dates[: -1])  ##注意_init_sequence_data的data_count和 _seq_data不一样!
        # Calculate bound index. #切割训练数据和测试数据的索引
        self.bound_index = int(self.data_count * self.training_data_ratio) # ##注意bound_index在_init_sequence_data的data_count和 _seq_data中也不一样!
        # Init scaled_x, scaled_y.
        scaled_data_x, scaled_data_y = [], []
        for index, date in enumerate(self.dates[: -1]):
            # Get current x, y. scaled_frames为一个字典
            x = [self.scaled_frames[code].iloc[index] for code in self.state_codes] # 返回"600318","600324","sh"三只股票的iloc的第[0]行数据的键值对
            # y 为第二天的数据,同时重建索引为0,1,2,3...
            y = [self.scaled_frames[code].iloc[index + 1] for code in self.state_codes] # 返回
            # Convert x, y to array.
            # 转换当天数据为张量，并reshape为一维向量
            x = np.array(x).reshape((1, -1)) #reshape成一维向量(1,len(dates)*len(state_codes)*5),相当于将拼接特征维度, 将sh上证指数和某一只股票的的特征维度拼接
            # 转换第二天数据为向量
            y = np.array(y)
            # Append x, y
            scaled_data_x.append(x) #
            scaled_data_y.append(y)
        # Convert list to array.
        self.data_x = np.array(scaled_data_x) # data_x形状为(len(dates),5*3) 2维
        self.data_y = np.array(scaled_data_y) #

    def _init_sequence_data(self):
        # Calculate data count.data_count=  总数据行数-1, 之所以要减去1,是因为要减去列名那行
        self.data_count = len(self.dates[: -1 - self.seq_length]) #注意_init_sequence_data的data_count和 _init_series_data的不一样! 切片的优先级高于四则运算 , 所以这里返回的是总日期长度-1再-5
        # Calculate bound index.
        self.bound_index = int(self.data_count * self.training_data_ratio) # 分割训练集和测试集的索引位置
        # Init seqs_x, seqs_y.
        scaled_seqs_x, scaled_seqs_y = [], []
        # Scale to valid dates.
        for date_index, date in enumerate(self.dates[: -1]): # 总共要循环dates列表长度次
            # Continue until valid date index.
            if date_index < self.seq_length: # 保证在索引位置大于5, 否则中断后面代码 ,后面的date_index 大于或等于seq_length, 这里为5
                continue
            data_x, data_y = [], []  #每次dates循环, 产生一个空列表data_x, data_y
            for index, code in enumerate(self.state_codes): # 每次循环一个dates, 在这次循环下再循环state_code长度次, 这里state_codes长度为3只股票, 其中包括1只为上证指数"sh"
                # Get scaled frame by code.
                scaled_frame = self.scaled_frames[code] #在某次循环内某个股票代码赋值给scaled_frame
                # Get instrument data x.
                instruments_x = scaled_frame.iloc[date_index - self.seq_length: date_index] # 从索引位置-5到索引位置5天的数据,每只股票5天的数据,形状为(5,特征dim)
                data_x.append(np.array(instruments_x)) # 往data_x空列表append某只股票代码5天的数据,一共循环state_codes长度==3 次, data_x里存在3个股票代码5天的数据
                # Get instrument data y.
                if index < date_index :#date_index取值为[0,1,2,3,4,5,6,7,...1342], index取值为[0,1,2],这里做的限制,初步判断应该是seq_length有可能会小于等于2
                    if date_index < self.bound_index:
                        # Get y, y is not at date index, but plus 1. (Training Set)
                        instruments_y = scaled_frame.iloc[date_index + 1]['close'] # 标签为索引位置+1后的一天的close dim维度, 形状为()
                    else:
                        # Get y, y is at date index. (Test Set)
                        instruments_y = scaled_frame.iloc[date_index + 1]['close'] # 标签为索引位置+1后的一天的close值, 每次取单个数,所以形状为()
                    data_y.append(np.array(instruments_y)) # 循环完3次之,列表中有三个,形状为(3,)
            # Convert list to array.
            data_x = np.array(data_x) #将列表转换成numpy数组,形状变为(3,5,特征dim)
            data_y = np.array(data_y) #将列表转换成numpy数组,形状为(3,)
            seq_x = []
            seq_y = data_y #index+1天的close价格
            # Build seq x, y.
            for seq_index in range(self.seq_length): # 循环5次
                seq_x.append(data_x[:, seq_index, :].reshape((-1))) #将三只股票代码同一天的数据取出来,列表对象seq_x
            # Convert list to array.
            seq_x = np.array(seq_x) # 将列表对象组装到numpy数组, 形状变为(5,3*特征dim),其中有一维度被展平,组合成的numpy数组为每天三个股票代码的特征dim的拼接, 一共产生5天的
            scaled_seqs_x.append(seq_x) #  列表组装
            scaled_seqs_y.append(seq_y) # 列表组装
        # Convert seq from list to array.
        self.seq_data_x = np.array(scaled_seqs_x) # 将列表转换成numpy数组 返回形状:(len(dates),5,3*特征dim),三维数组 ; 注意这是整个数据集
        self.seq_data_y = np.array(scaled_seqs_y) # 将列表转换成numpy数组 返回形状:(len(dates),3),二维数组; 注意这是整个数据集的标签

    def _init_data_indices(self):# 此方法中seq_data和series_data也都不一样
        # Calculate indices range.
        self.data_indices = np.arange(0, self.data_count)
        # Calculate train and eval indices.
        self.t_data_indices = self.data_indices[:self.bound_index] #返回训练数据集自然数集合
        self.e_data_indices = self.data_indices[self.bound_index:] #返回测试数据集自然数集合
        # Generate train and eval dates.
        self.t_dates = self.dates[:self.bound_index]# 返回训练集日期集合
        self.e_dates = self.dates[self.bound_index:]# 返回测试集日期集合

    def _origin_data(self, code, date):
        date_index = self.dates.index(date)  # list().index() python列表提供的方法, 返回索引位置,这里返回日期的索引位置
        return self.origin_frames[code].iloc[date_index]

    def _scaled_data_as_state(self, date):
        if not self.use_sequence:
            data = self.data_x[self.dates.index(date)] # data_x形状为(len(dates),5*3) 2维,所以这里返回某一个15特征维度的向量
            # 如果是混合交易，则在
            if self.mix_trader_state: # mix_trader_state指的是混合有指数的方式
                trader_state = self.trader.scaled_data_as_state()
                data = np.insert(data, -1, trader_state).reshape((1, -1))
            return data
        else:
            return self.seq_data_x[self.dates.index(date)]

    def reset(self, mode='train'):
        # Reset trader.
        self.trader.reset()
        # Reset iter dates by mode.
        self.iter_dates = iter(self.t_dates) if mode == 'train' else iter(self.e_dates)
        try:
            self.current_date = next(self.iter_dates)
            self.next_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Reset error, dates are empty.")
        # Reset baseline.
        self._reset_baseline()
        return self._scaled_data_as_state(self.current_date)

    def get_batch_data(self, batch_size=32):
        batch_indices = np.random.choice(self.t_data_indices, batch_size)
        #如果不是运用序列
        if not self.use_sequence:
            batch_x = self.data_x[batch_indices]
            batch_y = self.data_y[batch_indices]
        #如果是用序列
        else:
            batch_x = self.seq_data_x[batch_indices] # batch_x是5天窗口序列的数据
            batch_y = self.seq_data_y[batch_indices] # batch_y是5天后1天的label数据
        return batch_x, batch_y

    def get_test_data(self):
        if not self.use_sequence:
            test_x = self.data_x[self.e_data_indices]
            test_y = self.data_y[self.e_data_indices]
        else:
            test_x = self.seq_data_x[self.e_data_indices] # 这是整个测试集的所有输入序列
            test_y = self.seq_data_y[self.e_data_indices] # 这是整个测试集的输出序列,对应着每个输入序列后一天的close
        return test_x, test_y # 返回的是测试集

    def forward(self, stock_code, action_code): # 向前预测, 每隔一个时间步生成一个新数据
        # Check Trader.
        self.trader.remove_invalid_positions()
        self.trader.reset_reward()
        # Get stock data.
        stock = self._origin_data(stock_code, self.current_date)
        stock_next = self._origin_data(stock_code, self.next_date)
        # Execute transaction.
        action = self.trader.action_by_code(action_code)
        action(stock_code, stock, 100, stock_next)
        # Init episode status.
        episode_done = self.Running
        # Add action times.
        self.trader.action_times += 1
        # Update date if need.
        if self.trader.action_times == self.code_count:
            self._update_profits_and_baseline()
            try:
                self.current_date, self.next_date = self.next_date, next(self.iter_dates)
            except StopIteration:
                episode_done = self.Done
            finally:
                self.trader.action_times = 0
        # Get next state.
        state_next = self._scaled_data_as_state(self.current_date)
        # Return s_n, r, d, info.
        return state_next, self.trader.reward, episode_done, self.trader.cur_action_status

    def _update_profits_and_baseline(self):
        prices = [self._origin_data(code, self.current_date).close for code in self.codes]
        baseline_profits = np.dot(self.stocks_holding_baseline, np.transpose(prices)) - self.trader.initial_cash
        policy_profits = self.trader.profits
        self.trader.history_baselines.append(baseline_profits)
        self.trader.history_profits.append(policy_profits)

    def _reset_baseline(self):
        # Calculate cash piece.
        cash_piece = self.init_cash / self.code_count
        # Get stocks data.
        stocks = [self._origin_data(code, self.current_date) for code in self.codes]
        # Init stocks baseline.
        self.stocks_holding_baseline = [int(math.floor(cash_piece / stock.close)) for stock in stocks]

    @property # y_space
    def code_count(self): #刚好对应于#由_init_sequence_data返回 将列表转换成numpy数组 返回形状:(len(dates),4),二维数组
        return len(self.codes)

    @property
    def index_code_count(self):
        return len(self.index_codes)

    @property
    def state_code_count(self):
        return len(self.state_codes)

    @property
    def data_dim(self): # x_space 数据特征dim
        data_dim = self.state_code_count * self.scaled_frames[self.codes[0]].shape[1] # 这里的state_code_count包含了'sh'指数, 本行data_dim = 5*5, ..(len(dates),5,5*特征dim), 三维数组
        if not self.use_sequence:
            if self.mix_trader_state:
                data_dim += (2 + self.code_count)
        return data_dim
