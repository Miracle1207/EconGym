import copy
import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line, Grid, Timeline, Bar
from pyecharts.commons.utils import JsCode

"""
pyecharts 2.0.4

"""
tool_tip_formatter = """
function (params) { 
    let relVal = params[0] ? params[0].name : 'No Data'; 
    let year = 0;
    let firstValue = params[0] ? params[0].value : null;
    if (Array.isArray(firstValue)) {
        year = firstValue[0];
        firstValue = firstValue[1];
    }
    let numericFirstValue = Number(firstValue);
    if (isNaN(numericFirstValue)) {
        numericFirstValue = 0;
    }
    let formattedFirstValue = numericFirstValue.toFixed(2);
    relVal += `<br/>Year : <span style="float:right;">${year}</span>`;

    for (let i = 0; i < params.length; i++) { 
        let rawValue = params[i].value; 
        if (Array.isArray(rawValue)) { 
            rawValue = rawValue[1]; 
        } 
        let numericValue = Number(rawValue); 
        if (isNaN(numericValue)) { 
            numericValue = 0; 
        } 
        let formattedValue = numericValue.toFixed(2);
        relVal += `<br/>${params[i].marker}${params[i].seriesName} :   
                   <span style="float:right;">&ensp;${formattedValue}</span>`;
    } 
    return relVal; 
}
"""

# tool_tip_formatter_tiemline = """
# function (params) {
#     let result = '';
#     for (let i = 0; i < params.length; i++) {
#         let value = Number(params[i].value);
#         if (isNaN(value)) value = 0;
#         let formattedValue = value.toFixed(2);
#         result += `${params[i].marker}${params[i].seriesName}:
#                    <span style="float:right;">&ensp;${formattedValue}</span><br/>`;
#     }
#     return result;
# }
# """

tool_tip_formatter_tiemline = """
function (params) { 
    let relVal = params[0] ? params[0].name : ''; 
    let firstValue = params[0] ? params[0].value : null;
    if (Array.isArray(firstValue)) {
        year = firstValue[0];
        firstValue = firstValue[1];
    }
    let numericFirstValue = Number(firstValue);
    if (isNaN(numericFirstValue)) {
        numericFirstValue = 0;
    }
    let formattedFirstValue = numericFirstValue.toFixed(2);
    for (let i = 0; i < params.length; i++) { 
        let rawValue = params[i].value; 
        if (Array.isArray(rawValue)) { 
            rawValue = rawValue[1]; 
        } 
        let numericValue = Number(rawValue); 
        if (isNaN(numericValue)) { 
            numericValue = 0; 
        } 
        let formattedValue = numericValue.toFixed(2);
        relVal += `<br/>${params[i].marker}${params[i].seriesName} :   
                   <span style="float:right;">&ensp;${formattedValue}</span>`;
    } 
    return relVal; 
}
"""


def distribution_chart(house_data, category_data, house_age_dist, years):
    wealth_average = {}
    # wealth_labels = ['rich', 'middle-class', 'poor', 'mean']
    age_labels = ['<24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85 and older']
    for strategy_name, house_value in house_data.items():
        for per_year in range(len(house_value)):
            rich_sum = 0
            middle_sum = 0
            poor_sum = 0
            total = 0
            len_house = len(house_value[per_year])

            for per_house in range(len_house):
                per_house_value = house_data[strategy_name][per_year][per_house][0]
                per_house_category = category_data[strategy_name][per_year][per_house]

                total += per_house_value
                if per_house_category == 1:
                    rich_sum += per_house_value
                elif per_house_category == 2:
                    middle_sum += per_house_value
                else:
                    poor_sum += per_house_value

            rich_average = round(rich_sum / (len_house * 0.1), 3)
            middle_average = round(middle_sum / (len_house * 0.4), 3)
            poor_average = round(poor_sum / (len_house * 0.5), 2)
            mean = round(total / len_house, 3)

            if strategy_name not in wealth_average:
                wealth_average[strategy_name] = []

            wealth_average[strategy_name].append([rich_average, middle_average, poor_average, mean])

    house_age_count = {}
    for strategy_name, yearly_data in house_age_dist.items():
        house_age_count[strategy_name] = []
        for year_list in yearly_data:
            age_count = {label: 0 for label in age_labels}
            for age in year_list:
                if age in age_count:
                    age_count[age] += 1
            age_count['total'] = len(year_list)
            house_age_count[strategy_name].append(age_count)

    timeline = Timeline(init_opts=opts.InitOpts(width="1260px"))
    names = list(wealth_average.keys())
    max_years = max(len(years[name]) for name in names)
    for year_index in range(max_years):
        valid_names = [name for name in names if year_index < len(years[name])]

        if not valid_names:  # 如果没有有效的策略数据，跳过这个时间点
            continue

        bar_wealth = Bar()
        wealthy_data = [wealth_average[name][year_index][0] for name in valid_names]
        middle_data = [wealth_average[name][year_index][1] for name in valid_names]
        poor_data = [wealth_average[name][year_index][2] for name in valid_names]
        total_data = [wealth_average[name][year_index][3] for name in valid_names]

        bar_wealth.add_xaxis(valid_names)
        bar_wealth.add_yaxis("rich", wealthy_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("middle", middle_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("poor", poor_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("mean", total_data, label_opts=opts.LabelOpts(is_show=False))

        bar_wealth.set_global_opts(
            title_opts=opts.TitleOpts(title=f"Wealth Distribution at year {years[valid_names[0]][year_index]}",
                                      pos_right="50px"),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True)),
            tooltip_opts=opts.TooltipOpts(trigger='axis', formatter=JsCode(tool_tip_formatter_tiemline)),
            legend_opts=opts.LegendOpts(pos_right="50px", pos_top='6%', border_width=0, border_color='white'),
        )

        bar_population = Bar()
        age_labels = ['<24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85 and older'] + ['total']
        data_0 = [house_age_count[name][year_index]['<24'] for name in valid_names]  # <24
        data_1 = [house_age_count[name][year_index]['25-34'] for name in valid_names]  # 25-34
        data_2 = [house_age_count[name][year_index]['45-54'] for name in valid_names]  # 35-44
        data_3 = [house_age_count[name][year_index]['35-44'] for name in valid_names]  # 45-54
        data_4 = [house_age_count[name][year_index]['55-64'] for name in valid_names]  # 55-64
        data_5 = [house_age_count[name][year_index]['65-74'] for name in valid_names]  # 65-74
        data_6 = [house_age_count[name][year_index]['75-84'] for name in valid_names]  # 75-84
        data_7 = [house_age_count[name][year_index]['85 and older'] for name in valid_names]  # 85 and older
        data_8 = [house_age_count[name][year_index]['total'] for name in valid_names]  # total

        bar_population.add_xaxis(valid_names)

        # 添加每个年龄段作为单独的系列，并添加图例
        bar_population.add_yaxis(f"{age_labels[0]}", data_0, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[1]}", data_1, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[2]}", data_2, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[3]}", data_3, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[4]}", data_4, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[5]}", data_5, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[6]}", data_6, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[7]}", data_7, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[8]}", data_8, label_opts=opts.LabelOpts(is_show=False))

        bar_population.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Population Distribution at year {years[valid_names[0]][year_index]}"),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True)),
            tooltip_opts=opts.TooltipOpts(trigger='axis'),
            legend_opts=opts.LegendOpts(pos_left="50px", pos_top='6%', border_width=0, border_color='white'),
        )

        grid = Grid()
        grid.add(bar_wealth, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%"))
        grid.add(bar_population, grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%"))

        timeline.add(grid, f"Year {years[valid_names[0]][year_index]}")

    timeline.add_schema(play_interval=100, is_loop_play=False)


    return timeline


def age_wealth_dist_chart(house_data, category_data, house_age_dist, years, chart_name):
    wealth_average = {}
    # wealth_labels = ['rich', 'middle-class', 'poor', 'mean']
    for name, house_value in house_data.items():  # name指策略name,下同
        for per_year in range(len(house_value)):
            rich_sum = 0
            middle_sum = 0
            poor_sum = 0
            total = 0
            len_house = len(house_value[per_year])

            for per_house in range(len_house):
                per_house_value = house_data[name][per_year][per_house][0]
                per_house_category = category_data[name][per_year][per_house]

                total += per_house_value
                if per_house_category == 1:
                    rich_sum += per_house_value
                elif per_house_category == 2:
                    middle_sum += per_house_value
                else:
                    poor_sum += per_house_value

            rich_average = round(rich_sum / (len_house * 0.1), 3)
            middle_average = round(middle_sum / (len_house * 0.4), 3)
            poor_average = round(poor_sum / (len_house * 0.5), 2)
            mean = round(total / len_house, 3)

            if name not in wealth_average:
                wealth_average[name] = []

            wealth_average[name].append([rich_average, middle_average, poor_average, mean])

    age_labels = ['<24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85 and older']
    age_average = {}
    age_label_map = {label: index for index, label in enumerate(age_labels)}

    # 假设 house_data 和 category_data 已经定义好
    for name, house_value in house_data.items():
        for per_year in range(len(house_value)):
            # 初始化每个年龄段的总和
            age_sum = {label: 0 for label in age_labels}
            total = 0
            len_house = len(house_value[per_year])

            for per_house in range(len_house):
                # print(f"家庭{per_house}")
                per_house_value = house_data[name][per_year][per_house][0]
                per_house_age_category = house_age_dist[name][per_year][per_house]

                # 使用映射字典获取年龄分类的索引
                if per_house_age_category in age_label_map:
                    age_index = age_label_map[per_house_age_category]
                    age_label = age_labels[age_index]
                    age_sum[age_label] += per_house_value
                else:
                    raise ValueError(f"Invalid age category '{per_house_age_category}' found in data.")

                total += per_house_value

            # 计算每个年龄段的平均值
            age_averages = []
            for label in age_labels:
                age_proportion = 1 / len(age_labels)  # 假设均匀分布
                age_avg = round(age_sum[label] / (len_house * age_proportion), 3)
                age_averages.append(age_avg)
            # 加入均值
            mean = round(total / len_house, 3)
            age_averages.append(mean)

            if name not in age_average:
                age_average[name] = []

            age_average[name].append(age_averages)

    timeline = Timeline(init_opts=opts.InitOpts(width="1260px"))
    names = list(wealth_average.keys())
    max_years = max(len(years[name]) for name in names)
    for year_index in range(max_years):
        valid_names = [name for name in names if year_index < len(years[name])]

        if not valid_names:  # 如果没有有效的策略数据，跳过这个时间点
            continue

        bar_wealth = Bar()
        wealthy_data = [wealth_average[name][year_index][0] for name in valid_names]
        middle_data = [wealth_average[name][year_index][1] for name in valid_names]
        poor_data = [wealth_average[name][year_index][2] for name in valid_names]
        total_data = [wealth_average[name][year_index][3] for name in valid_names]

        bar_wealth.add_xaxis(valid_names)
        bar_wealth.add_yaxis("rich", wealthy_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("middle", middle_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("poor", poor_data, label_opts=opts.LabelOpts(is_show=False))
        bar_wealth.add_yaxis("mean", total_data, label_opts=opts.LabelOpts(is_show=False))

        bar_wealth.set_global_opts(
            title_opts=opts.TitleOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True)),
            tooltip_opts=opts.TooltipOpts(trigger='axis', formatter=JsCode(tool_tip_formatter_tiemline)),
            legend_opts=opts.LegendOpts(pos_right="50px", pos_top='6%', border_width=0, border_color='white'),
        )

        bar_population = Bar()
        age_labels = ['<24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85 and older'] + ['total']
        data_0 = [age_average[name][year_index][0] for name in valid_names]  # <24
        data_1 = [age_average[name][year_index][1] for name in valid_names]  # 25-34
        data_2 = [age_average[name][year_index][2] for name in valid_names]  # 35-44
        data_3 = [age_average[name][year_index][3] for name in valid_names]  # 45-54
        data_4 = [age_average[name][year_index][4] for name in valid_names]  # 55-64
        data_5 = [age_average[name][year_index][5] for name in valid_names]  # 65-74
        data_6 = [age_average[name][year_index][6] for name in valid_names]  # 75-84
        data_7 = [age_average[name][year_index][7] for name in valid_names]  # 85 and older
        data_8 = [age_average[name][year_index][8] for name in valid_names]  # total

        bar_population.add_xaxis(valid_names)

        # 添加每个年龄段作为单独的系列，并添加图例
        bar_population.add_yaxis(f"{age_labels[0]}", data_0, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[1]}", data_1, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[2]}", data_2, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[3]}", data_3, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[4]}", data_4, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[5]}", data_5, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[6]}", data_6, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[7]}", data_7, label_opts=opts.LabelOpts(is_show=False))
        bar_population.add_yaxis(f"{age_labels[8]}", data_8, label_opts=opts.LabelOpts(is_show=False))

        bar_population.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{chart_name} Distribution at year {years[valid_names[0]][year_index]}"),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True)),
            tooltip_opts=opts.TooltipOpts(trigger='axis'),
            legend_opts=opts.LegendOpts(pos_left="50px", pos_top='6%', border_width=0, border_color='white'),
        )

        grid = Grid()
        grid.add(bar_wealth, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%"))
        grid.add(bar_population, grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%"))

        timeline.add(grid, f"Year {years[valid_names[0]][year_index]}")

    # timeline.add_schema(play_interval=100, is_loop_play=False)
    # def set_global_opts(chart, chart_name, year, is_wealth_chart=False, is_inverse=False):
    #     # 公共配置
    #     common_opts = {
    #         "title_opts": opts.TitleOpts(
    #             title=f"{chart_name} at year {year}" if not is_wealth_chart else None,
    #             pos_left="50px",
    #             is_show=not is_wealth_chart,  # 财富分布图不显示标题
    #         ),
    #         "yaxis_opts": opts.AxisOpts(
    #             name="Wealth Class" if is_wealth_chart else "Age Range",
    #             name_location='start',
    #             name_gap='20',
    #         ),
    #         "tooltip_opts": opts.TooltipOpts(
    #             trigger='axis', formatter=JsCode(tool_tip_formatter_tiemline)
    #         ),
    #         "legend_opts": opts.LegendOpts(
    #             # is_show=False if is_wealth_chart else True,
    #             is_show=True,
    #             pos_left="50px",
    #             pos_top='6%',
    #             border_width=0,
    #             border_color='white',
    #         ),
    #     }
    #
    #     # 根据条件设置不同的 x 轴名称
    #     xaxis_name = (
    #         "Hours" if chart_name == "Households Work Hours"
    #         else "" if chart_name == "Households Reward"
    #         else "Dollars"
    #     )
    #     # 应用全局选项
    #     chart.set_global_opts(
    #         xaxis_opts=opts.AxisOpts(name=xaxis_name, is_inverse=is_inverse),
    #         **common_opts
    #     )
    #
    # # 创建时间轴
    # timeline = Timeline(init_opts=opts.InitOpts(width="1260px"))

    # strategy_names = list(age_average.keys())
    # max_year_len = max(len(years[name]) for name in strategy_names)
    #
    # for year_index in range(max_year_len):
    #     # 获取在当前年份有数据的策略
    #     valid_names = [
    #         name for name in strategy_names
    #         if year_index < len(wealth_average.get(name, [])) and wealth_average[name][year_index]
    #     ]
    #     if not valid_names:
    #         continue  # 没有任何策略在这个年份有数据，跳过
    #
    #     # -------- 财富分布图 --------
    #     bar_wealth = Bar()
    #
    #     for name in valid_names:
    #         data = wealth_average[name][year_index]
    #         bar_wealth.add_yaxis(name, data, label_opts=opts.LabelOpts(is_show=False))
    #
    #     bar_wealth.add_xaxis(wealth_labels)  # 这些是作为 y 轴的分类标签
    #     bar_wealth.reversal_axis()
    #     set_global_opts(bar_wealth, chart_name=chart_name, year=years[name][year_index], is_wealth_chart=True)
    #
    #     # -------- 年龄分布图 --------
    #     bar_population = Bar()
    #     bar_population.add_xaxis(age_labels + ['total'])  # 同样作为 y 轴分类标签
    #     for name in valid_names:
    #         data = age_average[name][year_index]
    #         bar_population.add_yaxis(name, data, label_opts=opts.LabelOpts(is_show=False))
    #     bar_population.reversal_axis()
    #
    #     set_global_opts(bar_population, chart_name=chart_name, year=years[name][year_index], is_inverse=True)
    #
    #     # -------- 组合进时间轴 --------
    #     grid = Grid()
    #     grid.add(bar_population, grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%"))
    #     grid.add(bar_wealth, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%"))
    #     timeline.add(grid, f"Year{year_index}")
    #

    # 设置时间轴播放选项
    timeline.add_schema(play_interval=100, is_loop_play=False)

    return timeline


def generate_timeline_chart(house_data, category_data, years, chart_name):
    average = {}
    for name, house_value in house_data.items():  # name指策略name,下同
        for per_year in range(len(house_value)):
            rich_sum = 0
            middle_sum = 0
            poor_sum = 0
            total = 0
            len_house = len(house_value[per_year])

            for per_house in range(len_house):
                per_house_value = house_data[name][per_year][per_house][0]

                per_house_category = category_data[name][per_year][per_house]

                total += per_house_value
                if per_house_category == 1:
                    rich_sum += per_house_value
                elif per_house_category == 2:
                    middle_sum += per_house_value
                else:
                    poor_sum += per_house_value

            rich_average = round(rich_sum / (len_house * 0.1), 3)
            middle_average = round(middle_sum / (len_house * 0.4), 3)
            poor_average = round(poor_sum / (len_house * 0.5), 2)
            total = round(total / len_house, 3)

            if name not in average:
                average[name] = []

            average[name].append([rich_average, middle_average, poor_average, total])

    timeline = Timeline(init_opts=opts.InitOpts(width="1260px"))
    strategy_names = list(average.keys())

    max_year_len = max(len(years[name]) for name in strategy_names)
    for year_index in range(max_year_len):
        valid_names = [name for name in strategy_names if year_index < len(years[name])]
        if not valid_names:
            continue

        bar = Bar()
        wealthy_data = [average[name][year_index][0] for name in valid_names]
        middle_data = [average[name][year_index][1] for name in valid_names]
        poor_data = [average[name][year_index][2] for name in valid_names]
        total_data = [average[name][year_index][3] for name in valid_names]
        bar.add_xaxis(valid_names)
        bar.add_yaxis("rich", wealthy_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("middle-class", middle_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("poor", poor_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("mean", total_data, label_opts=opts.LabelOpts(is_show=False))

        # 使用第一个有效策略的年份作为该时间点的标签
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=f"Average {chart_name} at year {years[valid_names[0]][year_index]}"),
            yaxis_opts=opts.AxisOpts(
                name=f"Average {chart_name}/Hour" if chart_name == "Households Work Hours" else f"Average {chart_name}",
                axislabel_opts=opts.LabelOpts(
                    formatter=JsCode("function (value) {return value.toExponential(1);}")
                ),
            ),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True)),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter=JsCode(tool_tip_formatter_tiemline)),
            legend_opts=opts.LegendOpts(pos_left="500px", pos_top='5%', border_width=0, border_color='white'),
        )

        timeline.add(bar, f"Year {years[valid_names[0]][year_index]}")

    timeline.add_schema(play_interval=100, is_loop_play=False)
    return timeline


# 传入两个参数 一个收入税/财产税，一个收入/财产
def generate_policy_chart(house_tax, house_data, years, chart_name):
    average = {}
    category_data = get_house_category(house_data)
    for strategy_name, house_value in house_data.items():
        for per_year in range(len(years[strategy_name])):
            rich_rate = 0
            middle_rate = 0
            poor_rate = 0
            total_rate = 0

            # 计算每一年里税占比
            house_number = len(house_value[per_year])
            for per_house in range(house_number):
                per_house_tax_value = house_tax[strategy_name][per_year][per_house][0]
                per_house_value = house_data[strategy_name][per_year][per_house][0]
                per_house_category = category_data[strategy_name][per_year][per_house]
                tax_rate = per_house_tax_value / (per_house_tax_value + per_house_value + 0.000001)

                total_rate += tax_rate
                if per_house_category == 1:
                    rich_rate += tax_rate
                elif per_house_category == 2:
                    middle_rate += tax_rate
                else:
                    poor_rate += tax_rate

            rich_average = round(rich_rate / (house_number * 0.1), 3)
            middle_average = round(middle_rate / (house_number * 0.4), 3)
            poor_average = round(poor_rate / (house_number * 0.5), 3)
            total_ = round(total_rate / house_number, 3)

            if strategy_name not in average:
                average[strategy_name] = []

            average[strategy_name].append([rich_average, middle_average, poor_average, total_])

    timeline = Timeline(init_opts=opts.InitOpts(width="1260px"))

    names = list(average.keys())
    for year_index in range(max(len(years[name]) for name in names)):
        valid_names = [name for name in names if year_index < len(years[name])]
        if not valid_names:  # 如果没有有效的策略数据，跳过这个时间点
            continue

        bar = Bar()
        wealthy_data = [average[name][year_index][0] for name in valid_names]
        middle_data = [average[name][year_index][1] for name in valid_names]
        poor_data = [average[name][year_index][2] for name in valid_names]
        total_data = [average[name][year_index][3] for name in valid_names]

        bar.add_xaxis(valid_names)
        bar.add_yaxis("rich", wealthy_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("middle", middle_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("poor", poor_data, label_opts=opts.LabelOpts(is_show=False))
        bar.add_yaxis("mean", total_data, label_opts=opts.LabelOpts(is_show=False))

        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=f"Average {chart_name} at year {years[valid_names[0]][year_index]}"),
            yaxis_opts=opts.AxisOpts(name=f"Average {chart_name}"),
            xaxis_opts=opts.AxisOpts(name="Strategy", axisline_opts=opts.AxisLineOpts(is_show=True),
                                     axislabel_opts={"interval": "0"}),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter=JsCode(tool_tip_formatter_tiemline)),
            # tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_left="500px", pos_top='5%', border_width=0, border_color='white'),
        )

        timeline.add(bar, f"Year {years[valid_names[0]][year_index]}")

    timeline.add_schema(play_interval=100, is_loop_play=False)
    return timeline


def generate_line_chart(chart_data, years, chart_name):
    # Initialize the Line chart object with custom initialization options
    line = Line(init_opts=opts.InitOpts(
        animation_opts=opts.AnimationOpts(
            animation=True,
            animation_delay=1000,
        )
    ))

    # Add data to the chart for each name
    for name, values in chart_data.items():
        # Set global options for the chart
        if chart_name in ["GDP", "Gov_Spending", "Gov_TaxIncome"]:
            line.set_global_opts(
                yaxis_opts=opts.AxisOpts(name=f"{chart_name}",
                                         axislabel_opts=opts.LabelOpts(formatter=JsCode(
                                             "function (value) {return value.toExponential(1);}"
                                         ))),
                title_opts=opts.TitleOpts(title=f"{chart_name} Over Time"),
                xaxis_opts=opts.AxisOpts(name="Year", min_=1, type_="value"),
                datazoom_opts=opts.DataZoomOpts(is_show=True),
                legend_opts=opts.LegendOpts(pos_left="220px", pos_top="25px", border_width=0, border_color='white'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', formatter=JsCode(tool_tip_formatter))
            )
        else:
            line.set_global_opts(yaxis_opts=opts.AxisOpts(name=f"{chart_name}"),
                                 title_opts=opts.TitleOpts(title=f"{chart_name} Over Time"),
                                 xaxis_opts=opts.AxisOpts(name="Year", min_=1, type_="value"),
                                 datazoom_opts=opts.DataZoomOpts(is_show=True),
                                 legend_opts=opts.LegendOpts(pos_left="220px", pos_top="25px", border_width=0,
                                                             border_color='white'),
                                 tooltip_opts=opts.TooltipOpts(trigger='axis', formatter=JsCode(tool_tip_formatter))

                                 )

        line.add_xaxis(years[name])
        line.add_yaxis(name, values, is_smooth=True, symbol_size=2,
                       label_opts=opts.LabelOpts(is_show=False))

    grid = Grid(init_opts=opts.InitOpts(width="1260px", height="600px"))
    grid.add(line, grid_opts=opts.GridOpts(pos_top="80px"))

    # Return the Line chart object
    return grid


def generate_ratio_chart(divisor_data, dividend_data, years, chart_name_1, chart_name_2):
    # Create Line charts
    divisor_line = Line()
    ratio_line = Line()

    # Adding data to the first chart
    for name, divisor_values in divisor_data.items():
        divisor_line.add_xaxis(years[name])
        divisor_line.add_yaxis(name, divisor_values, is_smooth=True, symbol_size=2,
                               label_opts=opts.LabelOpts(is_show=False))

    # Setting options for the first chart
    divisor_line.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{chart_name_1} And {chart_name_2} Over Time"),
        xaxis_opts=opts.AxisOpts(name="Year", min_=1, type_="value"),
        yaxis_opts=opts.AxisOpts(name=f"{chart_name_1}",
                                 axislabel_opts=opts.LabelOpts(formatter=JsCode(
                                     "function (value) {return value.toExponential(1);}"
                                 ))),
        tooltip_opts=opts.TooltipOpts(trigger='axis', formatter=JsCode(tool_tip_formatter)),
        datazoom_opts=[opts.DataZoomOpts(is_show=True, is_realtime=True, xaxis_index=[0, 1])],
        legend_opts=opts.LegendOpts(is_show=False),
    )

    # 第一个图为GDP的情况下，需要绘制GDP增速图。一般情况是GDP作为被除数，现在divisor_data,和dividend_data都是GDP
    if chart_name_1 == "GDP":
        # 计算每个策略的 GDP 增速
        for name, gdp_values in divisor_data.items():
            ratio = [0] * len(gdp_values)  # 将第一年增速初始化为 0
            for year in range(1, len(gdp_values)):
                # 计算增速数值*100,显示时加上%，下同
                ratio[year] = ((divisor_data[name][year] - divisor_data[name][year - 1]) /
                               divisor_data[name][year - 1]) * 100

            ratio_line.add_xaxis(years[name])
            ratio_line.add_yaxis(name, ratio, is_smooth=True, symbol_size=2,
                                 label_opts=opts.LabelOpts(is_show=False))
    else:
        for name, dividend_values in dividend_data.items():
            divisor_values = divisor_data[name]
            ratio = []
            for divisor, dividend in zip(divisor_values, dividend_values):
                if isinstance(divisor, list):
                    divisor = divisor[0]
                    if isinstance(divisor, list):
                        divisor = divisor[0]
    
                ratio.append((divisor / dividend) * 100)
            # ratio = [(divisor[0] / dividend) * 100 for divisor, dividend in zip(divisor_values, dividend_values)]
            ratio_line.add_xaxis(years[name])
            ratio_line.add_yaxis(name, ratio, is_smooth=True, symbol_size=2,
                                 label_opts=opts.LabelOpts(is_show=False))

    # Setting options for the second chart
    ratio_line.set_global_opts(
        xaxis_opts=opts.AxisOpts(name="Year", grid_index=1, min_=1, type_="value"),
        yaxis_opts=opts.AxisOpts(name=f"{chart_name_2}",
                                 axislabel_opts=opts.LabelOpts(formatter=JsCode(
                                     "function (value) {return value.toFixed(2)+'%';}"))
                                 ),

        legend_opts=opts.LegendOpts(pos_left="220px", pos_top="35px", border_width=0, border_color='white'),
        datazoom_opts=[opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1])],
    )

    # Combine charts into a Grid
    grid = Grid(init_opts=opts.InitOpts(width="1260x", height="1100px"))
    grid.add(divisor_line, grid_opts=opts.GridOpts(pos_top="7%", height="40%"))
    grid.add(ratio_line, grid_opts=opts.GridOpts(pos_top="53%", height="40%"))

    return grid


def get_total_tax(GDP_data, house_total_tax_data):
    total_tax_data = copy.deepcopy(GDP_data)
    for name, house_value in house_total_tax_data.items():
        for per_year in range(len(house_value)):
            house_total_tax_array = np.array(house_value[per_year])
            total_tax_data[name][per_year] = np.sum(house_total_tax_array)

    return total_tax_data


def get_house_category(house_wealth_data):
    house_category_data = copy.deepcopy(house_wealth_data)
    for name, house_value in house_wealth_data.items():
        for per_year in range(len(house_value)):
            wealth_array = np.array(house_value[per_year])
            sorted_array = np.sort(wealth_array, axis=0)[::-1]

            rich_threshold_value = sorted_array[int(0.1 * len(sorted_array)) - 1]
            middle_threshold_value = sorted_array[int(0.5 * len(sorted_array)) - 1]

            len_wealth_value = len(wealth_array)
            for per_house in range(len_wealth_value):
                per_house_value = house_wealth_data[name][per_year][per_house][0]
                if per_house_value >= rich_threshold_value:
                    house_category_data[name][per_year][per_house] = 1
                elif per_house_value >= middle_threshold_value:
                    house_category_data[name][per_year][per_house] = 2
                else:
                    house_category_data[name][per_year][per_house] = 3

    return house_category_data


def get_house_age_distribution(house_age_data, years, chart=False):
    house_age_distribution = copy.deepcopy(house_age_data)
    age_bins = [0, 24, 35, 45, 55, 65, 75, 85, np.inf]
    age_labels = ['<24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85 and older']
    for name, house_value in house_age_data.items():
        for per_year in range(len(house_value)):
            age_array = np.array(house_value[per_year])

            # 确保 age_array 是一维的
            age_array_flat = age_array.flatten()

            # 分类年龄
            age_categories = pd.cut(age_array_flat, bins=age_bins, labels=age_labels, right=False)

            # 将分类结果存储回 house_category_data 中
            for per_house in range(len(house_value[per_year])):
                house_age_distribution[name][per_year][per_house] = age_categories[per_house]

    return house_age_distribution
