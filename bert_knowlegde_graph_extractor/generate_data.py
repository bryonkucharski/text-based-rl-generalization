from ground_truth_textworld import *
import numpy as np
import textworld
import textworld.gym
import gym
import re
import sys
import glob
import requests
import json
import numpy as np
import copy

def clean_game_state(state):
    
    lines = state.split("\n")
    cur = [a.strip() for a in lines]
    cur = ' '.join(cur).strip().replace('\n', '').replace('---------', '')
    cur = re.sub("(?<=-\=).*?(?=\=-)", '', cur)
    cur = re.sub('[$_\\|/>]', '', cur)
    cur = cur.replace("-==-", '').strip()
    #cur = '. '.join([a.strip() for a in cur.split('.')])
    #cur = cur + '\n'
    return cur  



def generate_bert_triplet_data(games, seed, branching_depth):
    rng = np.random.RandomState(seed)
    dataset = []
    for game in tqdm(games):
        # Ignore the following commands.
        commands_to_ignore = ["look", "examine", "inventory"]

        request_infos = textworld.EnvInfos(admissible_commands=True, last_action = True, game = True, description=True, entities=True, facts = True, extras=["recipe","walkthrough"])
        env_id = textworld.gym.register_game(game, request_infos, max_episode_steps=10000)
        env = gym.make(env_id)

        _, infos = env.reset()
        walkthrough = infos["extra.walkthrough"]
        if walkthrough[0] != "inventory":  # Make sure we start with listing the inventory.
            walkthrough = ["inventory"] + walkthrough

        
        done = False
        cmd = "restart"  # The first previous_action is like [re]starting a new game.
        for i in range(len(walkthrough) + 1):
            obs, infos = env.reset()
            obs = infos["description"]  # `obs` would contain the banner and objective text which we don't want.

            # Follow the walkthrough for a bit.
            for cmd in walkthrough[:i]:
               
                obs, _, done, infos = env.step(cmd)
                state = "DESCRIPTION: "+ infos['description'] + "INVENTORY: "+ env.step('inventory')[0] + infos["extra.recipe"]
                state = clean_game_state(state)
                local_facts = process_local_facts(infos['game'], infos['facts'])
                serialized_facts = serialize_facts(local_facts)
                filtered_facts = filter_triplets(serialized_facts)

                dataset += [{
                    "game": os.path.basename(game),
                    "step": (i, 0),
                    "state": state,
                    "facts": filtered_facts
                    
                }]

            if done:
                break  # Stop collecting data if game is done.

            if i == 0:
                continue  # No random commands before 'inventory'

            # Then, take N random actions.
            for j in range(branching_depth):
                cmd = rng.choice([c for c in infos["admissible_commands"] if (c == "examine cookbook" or c.split()[0] not in commands_to_ignore)])
                obs, _, done, infos = env.step(cmd)
                if done:
                    break  # Stop collecting data if game is done.
                state = "DESCRIPTION: "+ infos['description'] + "INVENTORY: "+ env.step('inventory')[0] + infos["extra.recipe"]
                state = clean_game_state(state)
                local_facts = process_local_facts(infos['game'], infos['facts'])
                serialized_facts = serialize_facts(local_facts)
                filtered_facts = filter_triplets(serialized_facts)

                dataset += [{
                    "game": os.path.basename(game),
                    "step": (i, j + 1),
                    "state": state,
                    "facts": filtered_facts
                }]
    with open('data.json', 'w') as fp:
        json.dump(dataset, fp)


if __name__ == "__main__":
    #first download textworld games from  https://aka.ms/ftwp/dataset.zip
    
    PATH_TO_GAMES = ''        
    path = PATH_TO_GAMES + "/train/*.ulx"
    
    games = glob.glob(path)
    generate_bert_triplet_data(games,5154,5)
