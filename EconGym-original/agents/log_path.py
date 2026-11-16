

from pathlib import Path
import os

def make_logpath(args, n, task):
    file_name = generate_total_alg_name(args)
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path('./models') / task / str(n)

    if not model_dir.exists():
        curr_run = 'run1'

    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    os.makedirs(run_dir)
    file_name = task + "_" + str(n) + "_" + file_name

    return run_dir, file_name

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/parameters_setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def generate_total_alg_name(args):
    """
    Generates a concatenated string of algorithm names based on input arguments,
    excluding 'gov_alg' if any government-related algorithms are provided.
    """
    # Define the keys that need to be processed
    base_keys = ["house_alg", "gov_alg", "firm_alg", "bank_alg",
                 "central_bank_gov_alg", "tax_gov_alg", "pension_gov_alg"]
    
    # List to store the resulting parts of the total string
    result = []
    
    # Check if any of the special government-related keys are found
    gov_related_keys = ["central_bank_gov_alg", "tax_gov_alg", "pension_gov_alg"]
    gov_related_found = any(args.get(key, "") for key in gov_related_keys)
    
    # Loop through each key in the base_keys list
    for key in base_keys:
        value = args.get(key, "")  # Retrieve the value for the current key
        
        # If any of the special gov-related keys are found, skip "gov_alg"
        if gov_related_found and key == "gov_alg":
            continue
        
        if value:  # If the value is found (non-empty)
            name = key.split('_')[0] + "_" + value.split('_')[0]
            result.append(name)
    
    # Concatenate all the parts to form the final total_alg_name
    return "_".join(result)
