import os
import traceback
from typing import *

from statsmodels.formula import api
import numpy as np
import math
import datetime
import pandas as pd
import logging


# RSRS 择时 + 动量轮动 ETF 策略

# # 克隆自聚宽文章：https://www.joinquant.com/post/42198
# # 标题：RSRS择时ETF动量轮动_V17
# # 作者：Clarence.罗
#
# # 克隆自聚宽文章：https://www.joinquant.com/post/41804
# # 标题：RSRS择时ETF动量轮动_V16_fixd【送实盘版】
# # 作者：草根大树
#
# # 克隆自聚宽文章：https://www.joinquant.com/post/41718
# # 标题：多因子宽基ETF择时轮动改进版-高收益大资金低回撤
# # 作者：养家大哥
#
# # 标题：ETF动量轮动RSRS择时-V15.0，2023/3/23
# # 作者：养家大哥
#
# # 标题：动量ETF轮动RSRS择时-v16
# # 作者：杨德勇
# # v2 养家大哥的思路：
# # 趋势因子的特点是无法及时判断趋势的变向，往往趋势变向一段时间后才能跟上，
# # 巨大回撤往往就发生在这种时候。因此基于动量因子的一阶导数，衡量趋势的潜在变化速度，
# # 若变化速度过快则空仓，反之则按原计划操作。
# # 可以进一步发散，衡量动量因子的二阶导、三阶导等等，暂时只测试过一阶导，就是目前这个升级2版本。
#
# # 2023.05.22
# # 改正了 股票代码 错误的问题
# # 改正了 空仓时直接访问context.portfolio.position[stock]，系统报错的问题
# # 全面加注释
#
# from jqdata import *
# import numpy as np
# import pandas as pd
# from jqlib.technical_analysis import *

#     g.stock_pool = [
#         # ======== 大盘 ===================
#         '510300.XSHG',  # 沪深300ETF   13
#         '510050.XSHG',  # 上证50ETF
#         '159949.XSHE',  # 创业板50     25
#         '159928.XSHE',  # 中证消费ETF  10
#         #'510500.XSHG', # 500ETF       10
#         # '516160.XSHG', # 新能源ETF    28
#         # '512480.XSHG', # 半导体ETF    10
#         # '511260.XSHG', #十年国债      0.5
#         # '588050.XSHG', # 科创ETF      11
#         # '510880.XSHG', # 红利ETF        10.6
#         # '510180.XSHG', # 上证180 （用于替换上证50或沪深300，其与创业板有重合）
#         # '159915.XSHE', # 创业指数，替代创业500
#         # '159915.XSHE', # 创业板 ETF
#         # '512120.XSHG', # 医药50ETF
#         #'512100.XSHG', # 中证1000
#         # '159845.XSHE', # 中证1000
#     ]
#     # 备选池：用流动性和市值更大的50ETF分别代替宽指ETF，500与300ETF保留
#


class Context:
    def __init__(self, context):
        self.context = context

    @property
    def positions(self):
        return self.context.portfolio.positions

    @property
    def portfolio(self):
        return self.context.portfolio

    @property
    def previous_date(self) -> datetime.date:
        """获取前一个交易日"""
        return self.context.previous_date

    @property
    def hour(self) -> int:
        return self.context.current_dt.hour

    @property
    def minute(self) -> int:
        return self.context.current_dt.minute

    pass


class OrderStatus:
    pass


class Symbol(str):
    pass


def WR(*args, **kwargs) -> Tuple[pd.Series, pd.Series]:
    pass


def order_target_value():
    pass


def get_current_data():
    pass


# 聚宽 API 模拟补充
def attribute_history(
    security,
    count,
    unit="1d",
    fields=None,
    skip_paused=True,
    df=True,
    fq="pre",
) -> pd.DataFrame:
    fields = (
        fields
        if fields is not None
        else ["open", "close", "high", "low", "volume", "money"]
    )
    pass


def initialize():
    #     set_benchmark('510300.XSHG')
    #     set_slippage(FixedSlippage(0.001))
    #     set_order_cost(OrderCost(open_tax=0, close_tax=0.000, open_commission=0.0001, close_commission=0.0001, close_today_commission=0, min_commission=0), type='fund')

    #     #=============================================================================
    #     run_daily(my_trade_prepare, time='7:00', reference_security='000300.XSHG')
    #     run_daily(my_trade, time='9:30', reference_security='000300.XSHG')
    #     run_daily(my_sell2buy, time='9:35', reference_security='000300.XSHG')
    #     run_daily(check_lose, time='open', reference_security='000300.XSHG')
    #     # run_daily(print_trade_info, time='15:10', reference_security='000300.XSHG')
    #     #run_daily(pre_hold_check, time='11:25')
    #     run_daily(hold_check, time='11:27')
    #     #有其他人建议加入下午盘中判断，减少回撤率
    #     #run_daily(pre_hold_check, time='14:25')
    #     #run_daily(hold_check, time='14:27')
    pass


class RSRS_Momentum_Strategy:
    def __init__(self):
        self.ctx: Union[Context, None] = None

        self.stock_pool: List[Symbol] = []
        self.stock_num: int = 1  # 买入评分最高的前 stock_num 只股票
        self.momentum_day: int = 20  # 最新动量参考最近 momentum_day 的

        # === 大盘指数择时 ===
        self.ref_stock: Symbol = Symbol("000300.XSHG")  # 用 ref_stock 做择时计算的基础数据
        self.N: int = 18  # 计算最新斜率 slope, 拟合度 r2 参考最近 N 天
        self.M: int = 600  # 计算最新标准分 zscore, rsrs_score 参考最近 M 天
        self.K: int = 8  # 计算 zscore 斜率的窗口大小
        self.score_thr: float = -0.68  # rsrs 标准分指标阈值
        self.score_fall_thr: float = -0.43  # 当股票下跌趋势时候, 卖出阀值 rsrs
        self.idex_slope_raise_thr: float = 12  # 判断大盘指数强势的斜率门限

        # === 个股止盈止损 ===
        self.lossN: int = 20  # 止损 MA20 --- 60分钟
        self.lossFactor: float = 1.005  # 下跌止损的比例, 相对前一天的收盘价

        self.Motion_1diff: float = 19  # 股票前一天动量变化速度门限
        self.raiser_thr: float = 4.8  # 股票前一天上涨的比例门限

        self.biasN: int = 90  # 乖离动量的时间天数
        self.SwitchFactor: float = 1.04  # 换仓位的比例, 待换股相对当前持股的分数
        self.hold_stock: Union[
            Symbol, None
        ] = None  # 持仓的股票代码, 其 rsrs score 乘以 g.SwitchFactor 再排序判断是否轮动

        # === 测试其他值 ===
        self.Motion_1diff: float = 19  # 股票前一天动量变化速度门限
        self.raiser_thr: float = 4.8  # 股票前一天上涨的比例门限

        # === 指标缓存 ===
        (
            self.slope_series,
            self.rsrs_score_history,
        ) = self.initial_slope_series()  # 除去回测第一天的slope，避免运行时重复加入
        self.stock_motion = self.initial_stock_motion()  # 除去回测第一天的动量

        # === 信号 ===
        # 每日待操作的股票列表
        self.check_out_list: Union[Tuple[Symbol, float, float], None] = None
        # 买入/持有/卖出-操作信号
        self.timing_signal: Union[str, None] = None

    def initial_slope_series(self) -> Tuple[pd.Series, pd.Series]:
        """计算当前要用到的斜率和标准分数据"""
        n, m, k = self.N, self.M, self.K

        length = n + m + k
        data = attribute_history(self.ref_stock, length, "1d", ["high", "low", "close"])
        ols_res = [
            ols(data.low[i : i + n], data.high[i : i + n]) for i in range(length - n)
        ]
        df = pd.DataFrame(ols_res, columns=["intercept", "slope", "r2"])

        zscore_res = [
            zscore(df.slope[i + 1 : i + 1 + m]) * df.r2[i + m] for i in range(k)
        ]
        zscore_res = pd.Series(zscore_res)

        return df.slope, zscore_res

    def initial_stock_motion(self) -> Dict[Symbol, List[float]]:
        """计算股票池里面股票的动量因子"""
        bias_n = self.biasN
        momentum_day = self.momentum_day
        stock_pool = self.stock_pool

        stock_motion = {}
        for stock in stock_pool:
            motion_que = []
            data = attribute_history(stock, bias_n + momentum_day + 1, "1d", ["close"])
            data = data[:-1]

            bias = (data.close / data.close.rolling(bias_n).mean())[
                -momentum_day:
            ]  # 乖离因子
            score = (
                np.polyfit(np.arange(momentum_day), bias / bias[0], 1)[0].real * 10000
            )  # 乖离动量拟合

            motion_que.append(score)
            stock_motion[stock] = motion_que

        return stock_motion

    # 计算待买入的 <ETF 标的> 和 <择时信号>, 判断股票动量变化一阶导数, 如果变化太大, 则空仓
    # 本函数生成当日交易计划 - 主要就是根据历史数据计算当前相关的交易指标, 并生成:
    # 1. 交易信号
    # 2. 决定买入那个 ETF
    # 3. 决定是否卖出持有的 ETF
    def my_trade_prepare(self):
        print("———" * 10)

        hour = self.ctx.hour
        minute = self.ctx.minute
        # if hour == 9 and minute == 30:   # 9:30开盘时买入（标的根据昨天之前的数据算出来）

        # =============================================================================
        # 用大盘指数 ref_stock 的 RSRS 因子值和摆动指标作为<买入/持有/清仓>依据.
        timing_signal = self.get_timing_signal()
        self.timing_signal = timing_signal

        logging.info(f"今日大盘指数 {self.ref_stock} 择时信号: {timing_signal}")

        # =============================================================================
        # get_rank 返回了一只 RSRS 分最高的股票, 90天乖离率变化的斜率, 最近2天的涨跌幅. 斜率最大的排前面
        check_out_list = self.get_rank()
        logging.info(f"选了一只分数最高的ETF, 90天乖离率变化的斜率, 最近2天的涨跌幅 {check_out_list}")

        self.check_out_list = check_out_list

        # =============================================================================
        # 判断股票池中 第一支股票 动量变化一阶导数, 如果变化太大，则卖出
        cur_stock = check_out_list[0]
        cur_adr = check_out_list[2]  # 股票最近2天的涨跌幅
        cur_stock_motion = self.stock_motion[cur_stock]

        # 第一支股票 90 天乖离率的斜率的涨跌幅
        change_rate = cur_stock_motion[-1] - cur_stock_motion[-2]
        logging.info(f"选中的ETF, 最近2天的涨跌幅: {cur_adr}, 动量变化速度: {change_rate}")

        # =============================================================================
        if (change_rate > self.Motion_1diff) or (cur_adr > self.raiser_thr):
            timing_signal = "SELL"
            logging.info(f"选中的ETF, 由于涨跌: {cur_adr}, 动量变化: {change_rate:.2%}, 今日卖出")

        if timing_signal == "SELL":
            for stock in self.ctx.positions:
                self.send_message(f"卖出ETF [{stock}]")
        elif timing_signal == "BUY" or timing_signal == "KEEP":
            if check_out_list[0] not in self.ctx.positions:
                if len(self.ctx.positions) > 0:
                    stock_tmps = list(self.ctx.positions.keys())
                    self.send_message(
                        f"卖出ETF [{stock_tmps[0]}], 买入ETF [{check_out_list[0]}]"
                    )
                else:
                    self.send_message("买入ETF [%s]" % self.check_out_list[0])
        else:
            self.send_message("保持原来仓位")

    ## 动量因子：由收益率动量改为相对MA90均线的乖离动量
    def get_rank(self) -> Tuple[Symbol, float, float]:
        rank = []
        for stock in self.stock_pool:
            data = attribute_history(
                stock, self.biasN + self.momentum_day, "1d", ["close"]
            )
            bias = (data.close / data.close.rolling(self.biasN).mean())[
                -self.momentum_day :
            ]  # 乖离因子

            # The score is calculated as the slope of a linear regression line
            # between the values of the stock's "bias" over the past g.momentum_day days,
            # divided by the bias value on the first day.
            # 也即是 90天乖离率 变化速度 或 变化斜率
            score = (
                np.polyfit(np.arange(self.momentum_day), bias / bias[0], 1)[0].real
                * 10000
            )  # 乖离动量拟合

            # 计算股票最近两天收盘价的涨跌幅百分比，保存在变量 adr 中。
            adr = 100 * (data.close[-1] - data.close[-2]) / data.close[-2]  # 股票的涨跌幅度

            # 如果当前的股票代码 stock 等于程序中保存的持仓股票代码 g.hold_stock，那么将变量 raise_x 设为 g.SwitchFactor，否则设为 1。
            if stock == self.hold_stock:
                raise_x = self.SwitchFactor
            else:
                raise_x = 1

            # data = attribute_history(stock, g.momentum_day, '1d', ['close'])
            # score = np.polyfit(np.arange(g.momentum_day),data.close/data.close[0],1)[0].real # 乖离动量拟合
            # log.info("计算data.close[-1]=%f, data.close[-2]=%f,adr=%f"%(data.close[-1], data.close[-2], adr))

            # 保存<代码, 90天乖离率变化的斜率, 最近2天的涨跌幅>
            rank.append((stock, score * raise_x, adr))

            # 保存<代码, 90天乖离率变化的斜率>, 保留最近 5 天的就可以
            self.stock_motion[stock].append(score)
            if len(self.stock_motion[stock]) > 5:
                self.stock_motion[stock].pop(0)

        rank = [i for i in rank if math.isnan(i[1]) == False]
        rank.sort(key=lambda x: x[1], reverse=True)
        logging.info("各ETF排序 ", rank)
        return rank[0]

    # RSRS因子值和WR摆动指标 作为买入、持有和清仓依据，
    # 前版本还加入了移动均线的上行作为条件
    def get_timing_signal(self):
        n, m, k = self.N, self.M, self.K
        ref_stock = self.ref_stock

        # 18 天行情<最高,最低,收盘价>
        data = attribute_history(ref_stock, n, "1d", ["high", "low", "close"])

        # 光大斜率
        intercept, slope, r2 = ols(data.low, data.high)
        self.slope_series.append(slope)

        # 光大标准分
        rsrs_score = zscore(self.slope_series[-m:]) * r2
        self.rsrs_score_history.append(rsrs_score)

        # 光大斜率的斜率
        rsrs_slope = zscore_slope(self.rsrs_score_history[-k:])

        # 大盘指数收盘价趋势
        idex_slope = np.polyfit(np.arange(8), data.close[-8:], 1)[0].real

        # 只关注最近的数据, 历史数据可以清理掉
        self.slope_series.pop(0)
        self.rsrs_score_history.pop(0)

        # 通过摆动指数，提早知道趋势的变化，这种情况优先于RSRS
        WR2, WR1 = WR(
            [ref_stock],
            check_date=self.ctx.previous_date,
            N=21,
            N1=14,
            unit="1d",
            include_now=True,
        )

        # if WR1[g.ref_stock]<=2 and WR2[g.ref_stock] <=2: return "SELL"
        if WR1[ref_stock] >= 97 and WR2[ref_stock] >= 97:
            return "BUY"

        # 表示上升趋势快结束了，即将出现下跌
        if rsrs_slope < 0 and rsrs_score > 0:
            return "SELL"

        # 大盘下跌趋势过程中，不能买入
        if (idex_slope < 0) and (rsrs_slope > 0) and (rsrs_score < self.score_fall_thr):
            return "SELL"

        # 大盘上升过程当中，大胆买入
        if (idex_slope > self.idex_slope_raise_thr) and (rsrs_slope > 0):
            return "BUY"

        # 大盘可能上涨，这个时候可以买入
        if rsrs_score > self.score_thr:
            return "BUY"
        # elif(idex_slope > 5) : return "BUY"
        else:
            return "SELL"

    # 交易主函数，先确定ETF最强的是谁，然后再根据择时信号判断是否需要切换或者清仓
    def my_trade(self):
        timing_signal = self.timing_signal
        hour = self.ctx.hour
        minute = self.ctx.minute
        # if hour == 9 and minute == 30:   # 9:30开盘时买入（标的根据昨天之前的数据算出来）

        if timing_signal == "SELL":
            for stock in self.ctx.positions:
                position = self.ctx.positions[stock]
                self.close_position(position)
        elif timing_signal == "BUY" or timing_signal == "KEEP":
            self.adjust_position()
        else:
            pass

    # ===============================================================================
    # 4-2 交易模块-开仓
    # 买入指定价值的证券
    # 报单成功并成交(包括全部成交或部分成交, 此时成交量大于0)返回 True.
    # 报单失败或者报单成功但被取消(此时成交量等于0), 返回 False
    def open_position(self, security_code: str, value: int) -> bool:
        order = order_target_value(security_code, value)
        if order != None and order.filled > 0:
            return True

        return False

    # 4-3 交易模块-平仓
    # 卖出指定持仓
    # 报单成功并全部成交返回 True
    # 报单失败或者报单成功但被取消(此时成交量等于0), 或者报单非全部成交, 返回 False
    def close_position(self, position: int) -> bool:
        security_code = position.security
        order = order_target_value(security_code, 0)  # 可能会因停牌失败
        if order != None:
            if order.status == OrderStatus.held and order.filled == order.amount:
                return True

        return False

    # 4-4 交易模块-调仓
    def adjust_position(self):
        buy_stocks = self.check_out_list
        logging.info(f"按应买股票列表调仓: {buy_stocks}")

        for stock in self.ctx.positions.keys():
            if stock not in buy_stocks:
                logging.info(f"[{stock}]已不在应买股票列表中，调仓卖出")
                position = self.ctx.positions[stock]
                self.close_position(position)
                self.hold_stock = None
                return
            else:
                pass
                logging.info(f"[{stock}]已经持有，无需重复买入")

        position_count = len(self.ctx.positions)
        if self.stock_num > position_count:
            value = self.ctx.portfolio.cash / (self.stock_num - position_count)
            for stock in buy_stocks:
                if type(stock) != str:
                    logging.error(f"非法stock值={stock}")
                    continue

                if not (stock in self.ctx.positions.keys()):
                    if self.open_position(stock, value):
                        logging.info(f"[{stock}]在应买入列表中，调仓买入")
                        if len(self.ctx.positions) == self.stock_num:
                            self.hold_stock = stock
                            break

    # ===============================================================================
    def my_sell2buy(self):
        timing_signal = self.timing_signal
        hour = self.ctx.hour
        minute = self.ctx.minute
        # if hour == 9 and minute == 30:   # 9:30开盘时买入（标的根据昨天之前的数据算出来）
        if hour == 9:
            if timing_signal == "BUY" or timing_signal == "KEEP":
                logging.info(f"[{self.check_out_list}]启动buy_stocks特别买入")
                self.buy_stocks()
            else:
                pass

    # 跟 open_position 有啥区别？为啥单独来买一次？
    def buy_stocks(self):
        buy_stocks = self.check_out_list
        stock_num = self.stock_num
        position_count = len(self.ctx.positions)

        if stock_num > position_count:
            value = self.ctx.portfolio.cash / (stock_num - position_count)
            logging.warn(f"6-->买入股票 {buy_stocks}")
            for stock in buy_stocks:
                if type(stock) != str:
                    logging.error("6-->非法stock值", stock)
                    continue

                if not (stock in self.ctx.positions.keys()):
                    if self.open_position(stock, value):
                        if len(self.ctx.positions) == stock_num:
                            self.hold_stock = stock
                            break

    # 持仓检查，盘中动态止损：
    # 早盘结束后，若60分钟周期跌破MA20均线, 并且当前价格相对昨天没有上涨，则卖出
    def pre_hold_check(self):
        if self.ctx.positions:
            for stk in self.ctx.positions.keys():
                dt = attribute_history(stk, self.lossN + 2, "60m", ["close"])
                # dt['man'] = dt.close/dt.close.rolling(g.lossN).mean()
                # fix:for old pandas
                dt["man"] = dt.close / pd.rolling_mean(dt.close, g.lossN)
                if dt.man[-1] < 1.0:
                    stk_dict = self.ctx.positions[stk]
                    logging.info(f"盘中60分钟周期跌破MA20均线，止损卖出：{stk}")
                    self.send_message(f"盘中60分钟周期跌破MA20均线，止损卖出：{stk}")

    # 并且当前价格相对昨天没有上涨，则卖出
    def hold_check(self, ctx: Context):
        current_data = get_current_data()

        if ctx.positions:
            for stk in ctx.positions.keys():
                cur_stk = current_data[stk]

                yesterday_di = attribute_history(stk, 1, "1d", ["close"])
                dt = attribute_history(stk, self.lossN + 2, "60m", ["close"])
                dt["man"] = dt.close / dt.close.rolling(self.lossN).mean()

                # log.info("man=%0f, last_price=%0f, yester=%0f"%(dt.man[-1], current_data[stk].last_price*1.006, yesterday_di['close'][-1]))
                if (dt.man[-1] < 1.0) and (
                    cur_stk.last_price * self.lossFactor <= yesterday_di["close"][-1]
                ):
                    # if (dt.man[-1] < 1.0):
                    stk_dict = ctx.positions[stk]
                    logging.info("盘中60分钟周期跌破MA20均线，并且当前价格相对昨天没有上涨：{}".format(stk))
                    logging.info(
                        f"准备平仓，总仓位:{stk_dict.total_amount}, 可卖出：{stk_dict.closeable_amount}"
                    )
                    self.send_message("  盘中止损，卖出：{}".format(stk))
                    if stk_dict.closeable_amount:
                        order_target_value(stk, 0)
                        logging.info("  盘中止损卖出 ", stk)
                    else:
                        logging.info("  无法卖出 ", stk)

    def send_message(self, *args, **kwargs):
        logging.info(*args, **kwargs)


# 最小二乘法拟合, 得到: 截距/斜率/R^2
def ols(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
) -> Tuple[float, float, float]:
    model = api.ols(formula="y~x", data={"x": x, "y": y})
    result = model.fit()

    intercept = result.params[1]
    slope = result.params[1]
    r2 = result.rsquared

    return (intercept, slope, r2)


# 标准分计算: (slope - mean) / std
def zscore(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    mean = np.mean(data)
    std = np.std(data)
    return (data[-1] - mean) / std


# 对标准分在做一次最小二乘法拟合, 只要斜率
def zscore_slope(data: Union[np.ndarray, pd.Series]) -> float:
    y = data
    x = np.arange(len(data))
    intercept, slope, r2 = ols(x, y)
    return slope


# # 这个函数几乎没用
# # 止损函数
# def check_lose(context):
#     if len(context.portfolio.positions) < 1: return
#
#     for security_code in context.portfolio.positions.keys():
#         cost = context.portfolio.positions[security_code].avg_cost
#         price = context.portfolio.positions[security_code].price
#         ret = 100 * (price / cost - 1)
#
#         if ret <= -5:
#             order_target_value(security_code, 0)
#             print("！！！！！！触发止损信号: 标的={},标的价值={},浮动盈亏={}% ！！！！！！".format(security_code, format(value, '.2f'), format(ret, '.2f')))
#

# def print_trade_info(context):
#     # 打印当天成交记录
#     trades = get_trades()
#     for _trade in trades.values():
#         print('成交记录：' + str(_trade))
#     # 打印账户信息
#     print('———————————————————————————————————————分割线1————————————————————————————————————————')
