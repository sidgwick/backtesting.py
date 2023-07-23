# 斜率指标交易策略标准分策略
def WRX(HS300):
    data = HS300.copy()
    data.index.name = "date"
    data = data.reset_index()

    data["benchmark"] = (1 + data.close.pct_change(1).fillna(0)).cumprod()
    data["flag"] = 0  # 买卖标记
    data["position"] = 0  # 持仓标记
    position = 0  # 是否持仓，持仓：1，不持仓：0
    hold = 0
    cnt_n = 3
    for i in range(1, len(data)):
        # 开仓
        if data.loc[i, "cnt"] >= cnt_n:
            hold = 0
            data.loc[i, "flag"] = 1
            data.loc[i + 1, "position"] = 1
            position = 1
        # 平仓
        elif hold >= 3 and position == 1:
            hold = 0
            data.loc[i, "flag"] = -1
            data.loc[i + 1, "position"] = 0
            position = 0

        # 保持
        else:
            hold += 1
            data.loc[i + 1, "position"] = data.loc[i, "position"]

    # 假设卖出的动作是以收盘价完成的, 初始资金是 1 个单位
    # 这里先计算每日收盘价变动百分比(实际上也是基准收益)
    # 如果我们有持仓的话, 对应日期的资产应该也随着收盘价变动而变动.
    # 最终一个单位的资产乘以每日变化率, 就可以得到最终资产数量
    data["nav"] = (1 + data.close.pct_change(1).fillna(0) * data.position).cumprod()
    return data


def report(result, label="Strategy"):
    result = result.iloc[: len(result) - 1, :].copy(deep=True)
    last = result.shape[0] - 1
    num = result.flag.abs().sum() / 2
    nav = result.nav[last]

    print("交易次数 = ", num)
    print("策略净值为= ", nav)
    print("基准净值为= ", result.benchmark[last])

    def x(dt):
        t = dt.strftime("%Y-%m-%d")
        xt = datetime.datetime.strptime(t, "%Y-%m-%d")
        return xt

    result["datetime"] = [x(dt) for dt in result.date]
    ds = ColumnDataSource(result)

    fig = figure(
        x_axis_label="time",
        y_axis_label="profile",
        sizing_mode="stretch_width",
        height=300,
        x_axis_type="datetime",
    )

    fig.xaxis.formatter = DatetimeTickFormatter(days="%Y-%m-%d", months="%Y-%m")

    fig.line(
        x="datetime",
        y="nav",
        legend_label=label,
        line_width=2,
        color="red",
        source=ds,
    )

    fig.line(
        x=result.datetime,
        y=result.benchmark,
        color="yellow",
        legend_label="HS300",
        line_width=2,
    )

    show(fig)


result1 = WRX(data)
report(result1, label="RSRS1")
