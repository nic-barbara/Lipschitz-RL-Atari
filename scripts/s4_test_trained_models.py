from pathlib import Path

from liprl import utils
from liprl import adversarial as adv

# Setup
utils.startup_plotting()
dirpath = Path(__file__).resolve().parent
fpath = dirpath / "../results/params/"
apath = dirpath / "../results/attack-results/"
train_files = [str(f) for f in fpath.iterdir() if f.is_file()]

# Config params for making attack videos
# Attack sizes are the smallest (ish) required for these
# networks to lose the game. Can depend a lot on random seed.
video_config = {
    "Pong-v5": {
        "cnn": {
            "uniform": 21.0,
            "l2_pgd": 22.0,
            "linf_pgd": 0.42,
        },
        "lbdn": {
            "uniform": 35.0,
            "l2_pgd": 250.0,
            "linf_pgd": 1.95,
        }
    },
}

def evaluate_trained_model(load_path, 
                           video_config,
                           reward=False,
                           gameplay=False,
                           attacked_reward=False,
                           attacked_gameplay=False):
    """
    Plot results and generate gameplay videos for a single trained model.
    Also plots results for adversarial attacks.
    """
    
    # Load a trained model
    data = utils.load_params_configs(load_path)
    params, config, metrics = data["params"], data["config"], data["metrics"]
    
    # Only run for a few networks/Lipschitz bounds of interest   
    # Comment this out to analyse results for ALL models in save folder
    if not (config["seed"] == 1 and ((config["network"] == "cnn") or
            (config["network"] == "lbdn" and config["lipschitz"] == 10.0))):
        return None

    # Build environment and agent
    utils.seed_everything(config["seed"])
    env = utils.make_env(config["env_id"])
    agent = utils.load_agent(env, params, config)
        
    # Plot reward curve
    if reward:
        utils.plot_reward_curve(config, metrics)

    # Generate a gameplay video
    if gameplay:
        utils.make_gameplay_video(agent, config, save_frames=True)

    # Plot rewards after adversarial attacks
    if attacked_reward:
        
        data = adv.load_attack_results(apath / f"{config['fname']}.pickle")
        attack_metrics = data["attack_metrics"]
        
        for attacker in attack_metrics:
            adv.plot_attacked_rewards(config, attack_metrics[attacker])
        
    # Generate gameplay video with an attack
    if attacked_gameplay:
        env_video_config = video_config[config["env_id"]][config["network"]]
        for attacker in env_video_config:
            adv.attacked_gameplay_video(agent, config, attacker, 
                                        attack_size=env_video_config[attacker],
                                        save_frames=True)
        

for f in train_files:
    print(f.split("/")[-1])
    evaluate_trained_model(f, 
                           video_config,
                           reward=False, 
                           gameplay=True, 
                           attacked_reward=False,
                           attacked_gameplay=True)
