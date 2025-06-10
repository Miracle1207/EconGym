import argparse
import glob
import json
import pdb
import shutil

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
import os

from omegaconf import OmegaConf
from werkzeug.utils import secure_filename

from agents.ddpg_agent import ddpg_agent
from agents.ppo_agent import ppo_agent
from agents.real_data.real_data import real_agent
from agents.rule_based import rule_agent
from viz.chart import generate_line_chart, generate_ratio_chart, generate_timeline_chart, get_total_tax, \
    generate_policy_chart, get_house_category, distribution_chart, get_house_age_distribution, age_wealth_dist_chart
from env import EconomicSociety
from main import select_agent
from runner import Runner
from utils.seeds import set_seeds

# Define the directory containing the JSON files
DATA_FOLDER = 'viz/data'

years = {}
GDP_data = {}
# gov_reward_data = {}
social_welfare_data = {}
# per_gdp_data = {}
income_gini_data = {}
wealth_gini_data = {}
gov_spending_data = {}
# total_labor_data = {}
WageRate_data = {}
house_reward_data = {}

house_total_tax_data = {}
house_income_tax_data = {}  # 收入税/收入
house_wealth_tax_data = {}  # 资产税
house_wealth_data = {}
house_income_data = {}
house_consumption_data = {}
house_work_hour_data = {}
house_age_data = {}
house_pension_data = {}

charts = []
datasets = {}


# List and process files in both directories
def process_data():
    for file_name in os.listdir(DATA_FOLDER):
        # print(file_name)
        if file_name.endswith('_data.json'):
            strategy_name = file_name.replace('_data.json', '')
            file_path = os.path.join(DATA_FOLDER, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                years[strategy_name] = data.get('years', [])
                GDP_data[strategy_name] = data.get('GDP', [])
                income_gini_data[strategy_name] = data.get('income_gini', [])
                wealth_gini_data[strategy_name] = data.get('wealth_gini', [])
                social_welfare_data[strategy_name] = data.get('social_welfare', [])
                gov_spending_data[strategy_name] = data.get('gov_spending', [])

                house_income_tax_data[strategy_name] = data.get('house_income_tax', [])
                house_wealth_tax_data[strategy_name] = data.get('house_wealth_tax', [])
                house_total_tax_data[strategy_name] = data.get('house_total_tax', [])
                house_wealth_data[strategy_name] = data.get('house_wealth', [])
                house_income_data[strategy_name] = data.get('house_income', [])
                house_consumption_data[strategy_name] = data.get('house_consumption', [])
                house_work_hour_data[strategy_name] = data.get('house_work_hours', [])
                house_reward_data[strategy_name] = data.get('house_reward', [])
                WageRate_data[strategy_name] = data.get('WageRate', [])
                house_age_data[strategy_name] = data.get('house_age', [])
                # total_labor_data[strategy_name] = data.get('total_labor', [])
                house_pension_data[strategy_name] = data.get('house_pension', [])


datasets = {
    'GDP': GDP_data,
    'Social Welfare': social_welfare_data,
    'Income Gini': income_gini_data,
    'Wealth Gini': wealth_gini_data,
    'Government Spending': gov_spending_data,
    'Government Tax Income': house_total_tax_data,
    'Households Income Tax': house_income_tax_data,
    'Households Wealth Tax': house_wealth_tax_data,
    'Households Wealth': house_wealth_data,
    'Households Income': house_income_data,
    'Households Consumption': house_consumption_data,
    'Households Work Hours': house_work_hour_data,
    'Households Reward': house_reward_data,
    'Households WageRate': WageRate_data,
    'Households Age': house_age_data,
    'Households Distribution': house_wealth_data,
    'Households Pension': house_pension_data,

}


def clear_static_directory(directory):
    """Clear all HTML files in the specified directory."""
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
    except Exception as e:
        print(f"Error clearing directory: {e}")


def clear_static_models(directory):
    """Clear all model files in the specified directory."""
    for file_path in glob.glob(os.path.join(directory, "*")):
        os.remove(file_path)


app = Flask(
    __name__,
    template_folder="viz/templates"
)

MODEL_FOLDER = 'viz/models'  # Directory where uploaded files will be saved
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
DATA_FOLDER = 'viz/data'

app.secret_key = '123456789'

# Ensure the upload directory exists
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


@app.route('/')
def index():
    clear_static_models(MODEL_FOLDER)
    # 数据文件
    files = os.listdir(DATA_FOLDER)
    return render_template('index.html', files=files)


@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # 保存文件到指定目录
        file.save(os.path.join(app.config['DATA_FOLDER'], file.filename))
        return redirect(url_for('index'))


@app.route('/upload_model', methods=['POST'])
def upload_model():
    # Retrieve files from the request
    gov_actor = request.files.get('gov_actor')
    household_actor = request.files.get('household_actor')
    # 数据file
    files = os.listdir(DATA_FOLDER)
    # 模型file
    filenames = {}

    if not (gov_actor and household_actor):
        flash('No files selected', 'error')
        return render_template('index.html', files=files)

    if gov_actor.filename == '' or household_actor.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html', files=files)

    try:
        clear_static_models(MODEL_FOLDER)
        # Save the government model file
        if gov_actor:
            gov_filename = secure_filename(gov_actor.filename)
            gov_actor.save(os.path.join(app.config['UPLOAD_FOLDER'], 'gov_model.pt'))
            filenames['gov_actor'] = gov_filename.replace(".pt", "")

        # Save the household model file
        if household_actor:
            household_filename = secure_filename(household_actor.filename)
            household_actor.save(os.path.join(app.config['UPLOAD_FOLDER'], 'household_model.pt'))
            filenames['household_actor'] = household_filename.replace(".pt", "")

        flash('Files successfully uploaded', 'success')
    except Exception as e:
        flash(f'Error occurred: {str(e)}', 'error')
        return render_template('index.html', files=files)

    return render_template('index.html', filenames=filenames, files=files)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    os.remove(os.path.join(DATA_FOLDER, filename))
    return redirect(url_for('index'))


@app.route('/generate_data', methods=['POST'])
def generate_data():
    household_num = int(request.form.get('household_num'))
    gov_alg = request.form.get('gov_alg')
    house_alg = request.form.get('household_alg')
    house_model_path = "viz/models/household_model.pt"
    government_model_path = "viz/models/gov_model.pt"

    def select_agent(alg, agent_name):
        # Mapping of algorithms to agent constructors
        agent_constructors = {
            "real": real_agent,
            "mfrl": mfrl_agent,
            "bi_mfrl": bi_mfrl_agent,
            "ppo": ppo_agent,
            "ddpg": ddpg_agent,
            "maddpg": maddpg_agent,
            "rule_based": rule_agent,
            "aie": aie_agent,
            "bi_ddpg": bi_ddpg_agent,
            "us_federal": rule_agent,
            "saez": rule_agent
        }

        # Ensure the algorithm is supported
        if alg not in agent_constructors:
            raise ValueError("Wrong choice!")

        # Create agent using the appropriate constructor
        return agent_constructors[alg](env, yaml_cfg.Trainer, agent_name=agent_name)

    args = argparse.Namespace(
        config='default',  # 自定义配置文件名
        wandb=False,  # 是否启用 wandb
        test=False,  # 是否为测试模式
        br=False,  # 是否寻找最佳响应
        house_alg='ddpg',  # 家庭算法
        gov_alg='ddpg',  # 政府算法
        firm_alg='rule_based',  # 企业算法
        bank_alg='rule_based',  # 中央银行算法
        task='pension',  # 任务
        device_num=1,  # CUDA 设备数量
        households_n=100,  # 家庭数量
        seed=1,  # 随机种子
        hidden_size=128,  # 隐藏层大小
        q_lr=3e-4,  # Q网络学习率
        p_lr=3e-4,  # 策略网络学习率
        batch_size=64,  # 批量大小
        update_cycles=100,  # 更新周期
        update_freq=10,  # 更新频率
        initial_train=10,  # 初始训练步数
        aligned=False,
    )
    if gov_alg:
        args.gov_alg = gov_alg
    if house_alg:
        args.house_alg = house_alg
    if household_num:
        args.households_n = household_num

    try:
        # set signle thread
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        yaml_cfg = OmegaConf.load('cfg/default.yaml')
        yaml_cfg.Environment.Entities[1].entity_args.params.households_n = args.households_n
        yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
        yaml_cfg.Trainer["seed"] = args.seed
        yaml_cfg.Trainer["wandb"] = args.wandb
        yaml_cfg.Trainer["aligned"] = args.aligned
        yaml_cfg.Trainer["find_best_response"] = args.br

        # 调整参数
        yaml_cfg.Trainer["hidden_size"] = args.hidden_size
        yaml_cfg.Trainer["q_lr"] = args.q_lr
        yaml_cfg.Trainer["p_lr"] = args.p_lr
        yaml_cfg.Trainer["batch_size"] = args.batch_size
        yaml_cfg.Trainer["house_alg"] = args.house_alg
        yaml_cfg.Trainer["gov_alg"] = args.gov_alg

        if args.gov_alg == "saez" or args.gov_alg == "us_federal":
            yaml_cfg.Environment['env_core']['env_args']['tax_moudle'] = args.gov_alg
        set_seeds(args.seed, cuda=yaml_cfg.Trainer["cuda"])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
        env = EconomicSociety(yaml_cfg.Environment)

        # 选择代理
        house_agent = select_agent(args.house_alg, agent_name="household")
        gov_agent = select_agent(args.gov_alg, agent_name="government")
        firm_agent = select_agent(args.firm_alg, agent_name="firm")
        bank_agent = select_agent(args.bank_alg, agent_name="central_bank")

        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent,
                        firm_agent=firm_agent, bank_agent=bank_agent, )

        runner.viz_data(house_model_path, government_model_path)
        files = os.listdir(DATA_FOLDER)
        return render_template('index.html', files=files)

    except Exception as e:
        flash(f'Error occurred: {str(e)}', 'error')
        files = os.listdir(DATA_FOLDER)
        return render_template('index.html', files=files)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    # Output directory for charts
    # output_dir = 'static'
    user_ip = request.remote_addr
    output_dir = os.path.join('static', user_ip)
    os.makedirs(output_dir, exist_ok=True)
    clear_static_directory(output_dir)
    process_data()

    # Dictionary to store paths of generated charts
    chart_paths = {}
    selected_charts = request.form.get('charts')
    print('Selected charts:', selected_charts)
    if not selected_charts:
        return jsonify({'error': 'No charts selected'}), 400

    selected_charts = selected_charts.split(',')

    # Generate and save charts
    for title in selected_charts:
        chart_name = title.split('_', 1)[1]
        data = datasets.get(chart_name, [])
        category_data = get_house_category(datasets.get('Households Wealth'))  # poor rich
        house_age_dist = get_house_age_distribution(datasets.get('Households Age'), years)
        # years = datasets.get('years', [])

        if chart_name == "GDP":
            chart = generate_ratio_chart(data, data, years, chart_name, "GDP Growth Rate")

        elif chart_name == 'Government Spending':
            chart = generate_ratio_chart(data, datasets.get('GDP', []), years, chart_name,
                                         f"{chart_name} / GDP Ratio")
        # house_total_tax求出gov_tax_income
        elif chart_name == 'Government Tax Income':
            total_tax_data = get_total_tax(datasets.get('GDP', []), data)
            chart = generate_ratio_chart(total_tax_data, datasets.get('GDP', []), years, chart_name,
                                         f"{chart_name} / GDP Ratio")

        elif chart_name == "Households Income Tax":
            chart = generate_policy_chart(data, datasets.get("Households Income", []), years,
                                          f"{chart_name} / Income")

        elif chart_name == "Households Wealth Tax":
            chart = generate_policy_chart(data, datasets.get("Households Wealth", []), years,
                                          f"{chart_name} / Wealth")

        elif chart_name in ("Households Income", "Households Wealth", "Households Consumption", "Households Reward",
                            "Households Work Hours", 'Households Pension'):
            file_name = os.listdir(DATA_FOLDER)
            for file in file_name:
                if 'OLG' in file or 'pension' in file:
                    chart = age_wealth_dist_chart(data, category_data, house_age_dist, years, chart_name)
                else:
                    chart = generate_timeline_chart(data, category_data, years, chart_name)

        elif chart_name == 'Households Distribution':
            chart = distribution_chart(data, category_data, house_age_dist, years)

        else:
            chart = generate_line_chart(data, years, chart_name)

        if data:
            chart_filename = f'{title} chart.html'
            print(chart_filename)
            chart_path = os.path.join(output_dir, chart_filename)
            chart.render(chart_path)
            chart_paths[title] = chart_path

    return jsonify({'chart_paths': chart_paths})


if __name__ == '__main__':
    app.run(debug=True)
