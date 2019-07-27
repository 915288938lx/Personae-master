# coding=utf-8

from mongoengine import Document
from mongoengine import StringField, FloatField, DateTimeField


class Stock(Document): #类名即集合名
    # 股票代码
    code = StringField(required=True) #列名
    # 交易日
    date = DateTimeField(required=True)
    # 开盘价
    open = FloatField()
    # 最高价
    high = FloatField()
    # 最低价
    low = FloatField()
    # 收盘价
    close = FloatField()
    # 成交量
    volume = FloatField()
    # 成交金额
    amount = FloatField()
    # 涨跌幅
    p_change = FloatField()
    # 价格变动
    price_change = FloatField()
    # 5日均价
    ma5 = FloatField()
    # 10日均量
    ma10 = FloatField()
    # 20日均量
    ma20 = FloatField()
    # 5日均量
    v_ma5 = FloatField()
    # 10日均量
    v_ma10 = FloatField()
    # 20日均量
    v_ma20 = FloatField()
    # 换手率
    turnover = FloatField()

    meta = {
        # 索引，加快查询速度, indexes是mongoengine 专用的元数据保留词汇,这里建立了两个索引code和date
        'indexes': [
            'code',
            'date',
            ('code', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(code=self.code, date=self.date)) < 1 else None # 随时准备更新数据至最新

    def to_state(self):
        stock_dic = self.to_mongo() #查询结果转换为字典
        stock_dic.pop('_id')
        stock_dic.pop('code') #pop() 字典方法,删除键对应的值
        stock_dic.pop('date')
        return stock_dic.values() #返回剩下键的值得字典列表集合

    def to_dic(self):
        stock_dic = self.to_mongo()  #查询结果转换为字典
        stock_dic.pop('_id') # 删除_id 列
        return stock_dic.values() #返回剩下键的值的字典列表集合

    @classmethod
    def get_k_data(cls, code, start, end):
        return cls.objects(code=code, date__gte=start, date__lte=end).order_by('date') #date__gte = start pymongo,date__lte=end 提供的查询的格式, 条件时: date列大于start日期,小于end列, order_by(date) 升序

    @classmethod
    def exist_in_db(cls, code):
        return True if cls.objects(code=code)[:1].count() else False

















class Future(Document):
    # 合约代码
    code = StringField(required=True)
    # 交易日
    date = DateTimeField(required=True)
    # 开盘价
    open = FloatField()
    # 最高价
    high = FloatField()
    # 最低价
    low = FloatField()
    # 收盘价
    close = FloatField()
    # 成交量
    volume = FloatField()

    meta = {
        'indexes': [
            'code',
            'date',
            ('code', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(code=self.code, date=self.date)) < 1 else None

    def to_state(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        stock_dic.pop('code')
        stock_dic.pop('date')
        return stock_dic.values()

    def to_dic(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        return stock_dic.values()

    @classmethod
    def get_k_data(cls, code, start, end):
        return cls.objects(code=code, date__gte=start, date__lte=end).order_by('date')

    @classmethod
    def exist_in_db(cls, code):
        return True if cls.objects(code=code)[:1].count() else False
