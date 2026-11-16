import argparse
import glob
import json
import shutil

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os

from omegaconf import OmegaConf
from werkzeug.utils import secure_filename

from agents.rl.ddpg_agent import ddpg_agent
from agents.rl.ppo_agent import ppo_agent
from agents.rl.sac_agent import sac_agent
from agents.real_data.real_data import real_agent
from agents.rule_based.rules_core import rule_agent
from agents.data_based_agent import data_agent
from agents.behavior_cloning.bc_agent import bc_agent
from agents.llm.llm_agent import llm_agent
from viz.chart import generate_line_chart, generate_ratio_chart, generate_timeline_chart, get_total_tax, \
    generate_policy_chart, get_house_category, distribution_chart, get_house_age_distribution, age_wealth_dist_chart
from env import EconomicSociety
from runner import Runner
from utils.seeds import set_seeds

# Define the directory containing the JSON files
DATA_FOLDER = 'viz/data'

# Initialize dictionaries to store various economic data metrics
years = {}
GDP_data = {}
social_welfare_data = {}
income_gini_data = {}
wealth_gini_data = {}
gov_spending_data = {}
WageRate_data = {}
house_reward_data = {}

house_total_tax_data = {}
house_income_tax_data = {}  # Income tax / Income ratio
house_wealth_tax_data = {}  # Wealth tax
house_wealth_data = {}
house_income_data = {}
house_consumption_data = {}
house_work_hour_data = {}
house_age_data = {}
house_pension_data = {}

inflation_data = {}
base_interest_rate_data = {}
reserve_ratio_data = {}
deposit_rate_data = {}
lending_rate_data = {}


# Initialize lists/dictionaries for chart management
charts = []
datasets = {}


def process_data():
    """
    Process all JSON data files in the DATA_FOLDER.
    Extract economic metrics from each file and store them in corresponding dictionaries,
    using the strategy name (derived from filename) as the key.
    """
    for file_name in os.listdir(DATA_FOLDER):
        # Filter files ending with '_data.json' (standard data file naming convention)
        if file_name.endswith('_data.json'):
            # Extract strategy name by removing the '_data.json' suffix from the filename
            strategy_name = file_name.replace('_data.json', '')
            file_path = os.path.join(DATA_FOLDER, file_name)

            # Read and parse the JSON data file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Extract and store core economic system metrics
                years[strategy_name] = data.get('years', [])
                GDP_data[strategy_name] = data.get('GDP', [])
                income_gini_data[strategy_name] = data.get('income_gini', [])
                wealth_gini_data[strategy_name] = data.get('wealth_gini', [])
                social_welfare_data[strategy_name] = data.get('social_welfare', [])
                gov_spending_data[strategy_name] = data.get('gov_spending', [])

                # Extract and store household-specific metrics
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
                house_pension_data[strategy_name] = data.get('house_pension', [])

                inflation_data[strategy_name] = data.get('inflation_rate', [])
                base_interest_rate_data[strategy_name] = data.get('base_interest_rate', [])
                reserve_ratio_data[strategy_name] = data.get('reserve_ratio', [])
                deposit_rate_data[strategy_name] = data.get('deposit_rate', [])
                lending_rate_data[strategy_name] = data.get('lending_rate', [])

        


# Map data metric names to their corresponding data storage dictionaries
# Used for dynamic chart generation based on user selection
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
    'Inflation Rate': inflation_data,
    'Base Interest Rate': base_interest_rate_data,
    'Reserve Ratio': reserve_ratio_data,
    'Deposit Rate': deposit_rate_data,
    'Lending Rate': lending_rate_data,
}


def clear_static_directory(directory):
    """
    Clear all subdirectories and their contents in the specified static directory.
    Used to clean up old chart files before generating new ones.

    Args:
        directory (str): Path to the directory to be cleared
    """
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            # Only delete subdirectories (skip individual files if any)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
    except Exception as e:
        print(f"Error clearing directory: {e}")


def clear_static_models(directory):
    """
    Delete all model files (e.g., .pt files) in the specified model directory.
    Used to clean up old agent models before uploading new ones.

    Args:
        directory (str): Path to the model directory to be cleared
    """
    for file_path in glob.glob(os.path.join(directory, "*")):
        os.remove(file_path)


# Initialize Flask application
app = Flask(
    __name__,
    template_folder="viz/templates"  # Specify custom template directory for Flask
)

# Configure core application directories
MODEL_FOLDER = 'viz/models'  # Directory for storing uploaded agent model files
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
DATA_FOLDER = 'viz/data'  # Re-confirm data folder path (redundant but for clarity)

# Secret key required for Flask flash messages (for user feedback)
app.secret_key = '123456789'

# Create required directories if they don't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


@app.route('/')
def index():
    """
    Flask route for the homepage (root URL).
    Clears existing model files and displays available data files.

    Returns:
        Rendered homepage template with list of data files
    """
    clear_static_models(MODEL_FOLDER)
    # Get list of all files in the data directory to display on the frontend
    files = os.listdir(DATA_FOLDER)
    return render_template('index.html', files=files)


@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Flask route for uploading data files to the server.
    Handles file upload requests and saves files to the DATA_FOLDER.

    Returns:
        Redirect to homepage after successful upload, or error message for failures
    """
    # Check if a file part exists in the request
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']

    # Check if a file was actually selected
    if file.filename == '':
        return "No selected file", 400

    # Save the file if it exists and is valid
    if file:
        file.save(os.path.join(app.config['DATA_FOLDER'], file.filename))
        return redirect(url_for('index'))


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Flask route for uploading agent model files (government and household models).
    Clears old models, saves new ones, and provides user feedback via flash messages.

    Returns:
        Rendered homepage template with upload status (success/error)
    """
    # Retrieve uploaded model files from the request
    gov_actor = request.files.get('gov_actor')  # Government agent model file
    household_actor = request.files.get('household_actor')  # Household agent model file

    # Get lists of existing data and model files for frontend display
    files = os.listdir(DATA_FOLDER)
    filenames = {}

    # Validate that both model files were provided
    if not (gov_actor and household_actor):
        flash('No files selected', 'error')
        return render_template('index.html', files=files)

    # Validate that filenames are not empty
    if gov_actor.filename == '' or household_actor.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html', files=files)

    try:
        # Clear existing model files before saving new ones
        clear_static_models(MODEL_FOLDER)

        # Save government model file (rename to standard name for consistency)
        if gov_actor:
            gov_filename = secure_filename(gov_actor.filename)
            gov_actor.save(os.path.join(app.config['UPLOAD_FOLDER'], 'gov_model.pt'))
            filenames['gov_actor'] = gov_filename.replace(".pt", "")  # Remove .pt extension for display

        # Save household model file (rename to standard name for consistency)
        if household_actor:
            household_filename = secure_filename(household_actor.filename)
            household_actor.save(os.path.join(app.config['UPLOAD_FOLDER'], 'household_model.pt'))
            filenames['household_actor'] = household_filename.replace(".pt", "")  # Remove .pt extension for display

        # Show success message to the user
        flash('Files successfully uploaded', 'success')
    except Exception as e:
        # Show error message if any exception occurs during upload
        flash(f'Error occurred: {str(e)}', 'error')
        return render_template('index.html', files=files)

    # Return to homepage with uploaded filenames for display
    return render_template('index.html', filenames=filenames, files=files)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    """
    Flask route for deleting a specific data file from the DATA_FOLDER.

    Args:
        filename (str): Name of the file to be deleted (passed via URL)

    Returns:
        Redirect to homepage after successful deletion
    """
    os.remove(os.path.join(DATA_FOLDER, filename))
    return redirect(url_for('index'))


@app.route('/generate_data', methods=['POST'])
def generate_data():
    """
    Flask route for generating new economic simulation data.
    Reads user parameters (number of households, agent algorithms),
    initializes the simulation environment and agents, runs the simulation,
    and saves the output data.

    Returns:
        Rendered homepage template with generated data files (or error message)
    """
    # Get user input parameters from the request form
    household_num = int(request.form.get('household_num'))  # Number of households in the simulation
    gov_alg = request.form.get('gov_alg')  # Algorithm used for government agent
    house_alg = request.form.get('household_alg')  # Algorithm used for household agents

    # Paths to saved model files (for pre-trained agents)
    house_model_path = "viz/models/household_model.pt"
    government_model_path = "viz/models/gov_model.pt"

    def select_agent(alg, agent_name):
        """
        Helper function to select and initialize the appropriate agent based on the specified algorithm.

        Args:
            alg (str): Name of the agent algorithm (e.g., 'ppo', 'rule_based')
            agent_name (str): Name of the agent (e.g., 'households', 'government')

        Returns:
            Initialized agent instance

        Raises:
            ValueError: If the specified algorithm is not supported
        """
        # Mapping of algorithm names to their corresponding agent constructor functions
        agent_constructors = {
            "real": real_agent,
            "ppo": ppo_agent,
            "sac": sac_agent,
            "rule_based": rule_agent,
            "bc": bc_agent,
            "llm": llm_agent,
            "data_based": data_agent,
            "ddpg": ddpg_agent,
            "saez": rule_agent,  # Saez tax policy (implemented as a rule-based agent)
            "us_federal": rule_agent  # US Federal tax policy (implemented as a rule-based agent)
        }

        # Validate that the algorithm is supported
        if alg not in agent_constructors:
            raise ValueError("Wrong choice!")

        # Create and return the agent instance
        return agent_constructors[alg](env, yaml_cfg.Trainer, agent_name=agent_name)

    # Initialize default command-line arguments for the simulation
    # These can be overridden by user input from the frontend
    args = argparse.Namespace(
        config='default',  # Name of the default configuration file
        wandb=False,  # Disable Weights & Biases logging (for experiment tracking)
        test=False,  # Run in training mode (not test mode)
        br=False,  # Disable "best response" calculation
        house_alg='ddpg',  # Default household agent algorithm
        gov_alg='ddpg',  # Default government agent algorithm
        firm_alg='rule_based',  # Default firm agent algorithm (rule-based)
        bank_alg='rule_based',  # Default central bank algorithm (rule-based)
        task='pension',  # Simulation task focus (pension system)
        device_num=1,  # Number of CUDA devices to use (1 for single GPU)
        households_n=100,  # Default number of households
        seed=1,  # Random seed for reproducibility
        hidden_size=128,  # Size of hidden layers in neural network agents
        q_lr=3e-4,  # Learning rate for Q-networks (in RL agents)
        p_lr=3e-4,  # Learning rate for policy networks (in RL agents)
        batch_size=64,  # Batch size for training (in RL agents)
        update_cycles=100,  # Number of update cycles for training
        update_freq=10,  # Frequency of network updates (steps per update)
        initial_train=10,  # Initial steps of training before evaluation
        aligned=False,  # Disable aligned training (for multi-agent coordination)
    )

    # Override default arguments with user input from the frontend
    if gov_alg:
        args.gov_alg = gov_alg
    if house_alg:
        args.house_alg = house_alg
    if household_num:
        args.households_n = household_num

    try:
        # Configure environment variables for single-threaded operation
        # Prevents potential conflicts with multi-threaded libraries (e.g., NumPy)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        # Load the base simulation configuration from the YAML file
        yaml_cfg = OmegaConf.load('cfg/base_config.yaml')

        # Update configuration with user-specified parameters
        yaml_cfg.Environment.Entities[1].entity_args.params.households_n = args.households_n
        yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
        yaml_cfg.Trainer["seed"] = args.seed
        yaml_cfg.Trainer["wandb"] = args.wandb
        yaml_cfg.Trainer["aligned"] = args.aligned
        yaml_cfg.Trainer["find_best_response"] = args.br

        # Update RL agent training parameters in the configuration
        yaml_cfg.Trainer["hidden_size"] = args.hidden_size
        yaml_cfg.Trainer["q_lr"] = args.q_lr
        yaml_cfg.Trainer["p_lr"] = args.p_lr
        yaml_cfg.Trainer["batch_size"] = args.batch_size
        yaml_cfg.Trainer["house_alg"] = args.house_alg
        yaml_cfg.Trainer["gov_alg"] = args.gov_alg

        # Special case: Set tax module based on government algorithm (for rule-based tax policies)
        if args.gov_alg == "saez" or args.gov_alg == "us_federal":
            yaml_cfg.Environment['env_core']['env_args']['tax_moudle'] = args.gov_alg

        # Set random seeds for reproducibility (across Python, NumPy, PyTorch)
        set_seeds(args.seed, cuda=yaml_cfg.Trainer["cuda"])

        # Configure which CUDA device to use (for GPU acceleration)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

        # Initialize the economic simulation environment
        env = EconomicSociety(yaml_cfg.Environment)

        # Initialize agents using the selected algorithms
        house_agent = select_agent(args.house_alg, agent_name="households")
        gov_agent = select_agent(args.gov_alg, agent_name="government")
        firm_agent = select_agent(args.firm_alg, agent_name="firm")
        bank_agent = select_agent(args.bank_alg, agent_name="central_bank")

        # Initialize the simulation runner to manage the interaction between agents and environment
        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent,
                        firm_agent=firm_agent, bank_agent=bank_agent)

        # Run the simulation and generate visualization data (saves to DATA_FOLDER)
        runner.viz_data(house_model_path, government_model_path)

        # Get updated list of data files to display on frontend
        files = os.listdir(DATA_FOLDER)
        return render_template('index.html', files=files)

    except Exception as e:
        # Show error message if simulation fails
        flash(f'Error occurred: {str(e)}', 'error')
        files = os.listdir(DATA_FOLDER)
        return render_template('index.html', files=files)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """
    Flask route for generating visual charts based on processed simulation data.
    Creates a user-specific directory to store charts, cleans old charts,
    and generates selected chart types using the preprocessed data.

    Returns:
        JSON response with paths to generated charts (for frontend rendering)
    """
    # Get user's IP address to create a unique directory for their charts (avoids conflicts)
    user_ip = request.remote_addr
    output_dir = os.path.join('static', user_ip)

    # Create the user-specific directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear old charts in the user's directory to avoid outdated visuals
    clear_static_directory(output_dir)

    # Process raw JSON data into structured metrics (populates the `datasets` dictionary)
    process_data()

    # Dictionary to store paths of generated charts (key: chart title, value: file path)
    chart_paths = {}

    # Get list of selected chart types from the frontend request
    selected_charts = request.form.get('charts')
    print('Selected charts:', selected_charts)  # Debug log to verify selected charts

    # Return error if no charts are selected
    if not selected_charts:
        return jsonify({'error': 'No charts selected'}), 400

    # Split the comma-separated string of selected charts into a list
    selected_charts = selected_charts.split(',')

    # Generate each selected chart based on its type
    for title in selected_charts:
        # Extract the core chart name (removes prefix if present, e.g., "1_GDP" → "GDP")
        chart_name = title.split('_', 1)[1]

        # Get the corresponding data from the preprocessed datasets
        data = datasets.get(chart_name, [])

        # Precompute auxiliary data needed for specific chart types:
        # 1. Household category data (e.g., "poor", "middle", "rich" based on wealth)
        category_data = get_house_category(datasets.get('Households Wealth'))
        # 2. Household age distribution data (count of households per age group over years)
        house_age_dist = get_house_age_distribution(datasets.get('Households Age'), years)

        # Generate chart based on the metric type (uses specialized chart functions)
        if chart_name == "GDP":
            # GDP: Generate ratio chart to show GDP growth rate (data vs data, with years as x-axis)
            chart = generate_ratio_chart(data, data, years, chart_name, "GDP Growth Rate")

        elif chart_name == 'Government Spending':
            # Government Spending: Show ratio of spending to GDP (spending data vs GDP data)
            chart = generate_ratio_chart(data, datasets.get('GDP', []), years, chart_name,
                                         f"{chart_name} / GDP Ratio")

        elif chart_name == 'Government Tax Income':
            # Government Tax Income: First calculate total tax, then show tax-to-GDP ratio
            total_tax_data = get_total_tax(datasets.get('GDP', []), data)
            chart = generate_ratio_chart(total_tax_data, datasets.get('GDP', []), years, chart_name,
                                         f"{chart_name} / GDP Ratio")

        elif chart_name == "Households Income Tax":
            # Household Income Tax: Show ratio of income tax to household income
            chart = generate_policy_chart(data, datasets.get("Households Income", []), years,
                                          f"{chart_name} / Income")

        elif chart_name == "Households Wealth Tax":
            # Household Wealth Tax: Show ratio of wealth tax to household wealth
            chart = generate_policy_chart(data, datasets.get("Households Wealth", []), years,
                                          f"{chart_name} / Wealth")

        elif chart_name in ("Households Income", "Households Wealth", "Households Consumption",
                            "Households Reward", "Households Work Hours", 'Households Pension'):
            # Household-specific metrics: Use OLG/pension-aware chart if data is from OLG simulation
            # Check if any data file indicates an OLG (Overlapping Generations) or pension scenario
            file_name_list = os.listdir(DATA_FOLDER)
            olg_or_pension_flag = any(('OLG' in file) or ('pension' in file) for file in file_name_list)

            if olg_or_pension_flag:
                # OLG/pension data: Use age-wealth distribution chart (shows metrics by age group)
                chart = age_wealth_dist_chart(data, category_data, house_age_dist, years, chart_name)
            else:
                # Standard data: Use timeline chart (shows metrics over time by household category)
                chart = generate_timeline_chart(data, category_data, years, chart_name)

        elif chart_name == 'Households Distribution':
            # Household Distribution: Show distribution of households by wealth/age
            chart = distribution_chart(data, category_data, house_age_dist, years)
        elif chart_name in [
            'Inflation Rate',
            'Base Interest Rate',
            'Reserve Ratio',
            'Deposit Rate',
            'Lending Rate'
        ]:
            chart = generate_line_chart(data, years, chart_name)
        else:
            # Default case: Use basic line chart for metrics without specialized formatting
            chart = generate_line_chart(data, years, chart_name)

        # Save the generated chart to the user's directory if data exists
        if data:
            chart_filename = f'{title} chart.html'  # Unique filename for each chart
            print(chart_filename)  # Debug log to verify filename
            chart_path = os.path.join(output_dir, chart_filename)
            chart.render(chart_path)  # Render the chart to an HTML file
            chart_paths[title] = chart_path  # Store path for frontend access

    # Return chart paths as JSON so frontend can load and display the charts
    return jsonify({'chart_paths': chart_paths})


if __name__ == '__main__':
    """
    Entry point for the Flask application.
    Runs the app in debug mode (auto-reloads on code changes, shows errors in browser).
    """
    app.run(debug=True)
