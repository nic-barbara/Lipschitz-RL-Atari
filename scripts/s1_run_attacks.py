import numpy as np
from pathlib import Path

from liprl import utils
from liprl import adversarial as adv

# Setup
utils.startup_plotting()
dirpath = Path(__file__).resolve().parent
fpath = dirpath / "../results/params/"
savepath = dirpath / "../results/attack-results/"
files = [str(f) for f in fpath.iterdir() if f.is_file()]

# Attack params
attack_config = {
    "Pong-v5": {
        "uniform": np.linspace(0, 50, 21),
        "l2_pgd": np.linspace(0, 200, 21),
        "linf_pgd": np.linspace(0, 3, 21),
    },
}


def attack_trained_model(load_path, savepath, attack_config):
    
    # Load params for a trained model
    data = utils.load_params_configs(load_path)
    params, config = data["params"], data["config"]

    # Build environment and agent
    utils.seed_everything(config["seed"])
    env = utils.make_env(config["env_id"])
    agent = utils.load_agent(env, params, config)

    # Loop attackers and get attacked rewards
    env_config = attack_config[config["env_id"]]
    attack_results = {k: [] for k in env_config.keys()}
    for attacker in env_config:
        attack_range = env_config[attacker]
        attack_results[attacker] = adv.get_attacked_rewards(agent,
                                                            config,
                                                            attacker,
                                                            attack_range)
        attack_results[attacker]["attacker"] = attacker
        
    # Save results
    adv.save_attack_results(savepath, config, attack_results)
   

for f in files:
    print(f.split("/")[-1])
    attack_trained_model(f, savepath, attack_config)
