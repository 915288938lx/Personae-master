# coding=utf-8

import tushare as ts
import logging

from base.model.document import Stock
from helper.args_parser import stock_spider_parser


class StockSpider(object):
    def __init__(self, code, start="2008-01-01", end="2019-07-19"):
        self.code = code
        self.start = start
        self.end = end

    def crawl(self):
        stock_frame = ts.get_k_data(code=self.code, start=self.start, end=self.end, retry_count=30)
        for index in stock_frame.index: # stock_index是从0到length(stock_frame)的整数
            stock_series = stock_frame.loc[index] # 某一行的数据
            stock_dict = stock_series.to_dict() # pandas提供的字典化方法, 返回{"date":"2018-08-01","open":5234,"close":5272,"high":5298,"volume":7665600,"code":"sh"}
            stock = Stock(**stock_dict) # 将字典的键和Stock类中各个字段对应起来进行组装
            stock.save_if_need() # 储存进数据库
        logging.warning("Finish crawling code: {}, items count: {}".format(self.code, stock_frame.shape[0]))


def main(args):
    codes = args.codes
    # codes = ['sh']
    for _code in codes:
        StockSpider(_code, args.start, args.end).crawl()


if __name__ == '__main__':
    main(stock_spider_parser.parse_args())
