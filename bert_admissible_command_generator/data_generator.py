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
    cur = cur.replace("\\", "").strip()
    return cur  


def generate_data(games, seed, branching_depth):
    rng = np.random.RandomState(seed)
    dataset = []
    seen_states = set()
    for game in tqdm(games):
        # Ignore the following commands.
        commands_to_ignore = ["look", "examine", "inventory"]

        request_infos = textworld.EnvInfos(admissible_commands=True, last_action = True, game = True,inventory=True, description=True, entities=True, facts = True, extras=["recipe","walkthrough","goal"])
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
                state = "DESCRIPTION: "+ infos['description'] + " INVENTORY: "+ infos['inventory']
                state = clean_game_state(state)

                if state not in seen_states:

                    acs = infos['admissible_commands']
                    for ac in acs[:]:
                        if ac.startswith('examine') and ac != 'examine cookbook' or ac == 'look' or ac == 'inventory':
                            acs.remove(ac)
                    data = acs
                    data_name = 'admissible_commands'
                    

                    dataset += [{
                        "game": os.path.basename(game),
                        "step": (i, 0),
                        "state": state,
                        data_name : data
                    }]

                    seen_states.add(state)

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
                state = "DESCRIPTION: "+ infos['description'] + " INVENTORY: "+ infos['inventory']
                state = clean_game_state(state)
                if state not in seen_states:

                    acs = infos['admissible_commands']
                    for ac in acs[:]:
                        if (ac.startswith('examine') and ac != 'examine cookbook') or ac == 'look' or ac == 'inventory':
                            acs.remove(ac)
                    data = acs
                    data_name = 'admissible_commands'

                    dataset += [{
                        "game": os.path.basename(game),
                        "step": (i, j),
                        "state": state,
                        data_name : data
                    }]
                    seen_states.add(state)

    with open('data.json', 'w') as fp:
        json.dump(dataset, fp)


if __name__ == "__main__":

    #first download textworld games from  https://aka.ms/ftwp/dataset.zip
    
    PATH_TO_GAMES = ''        
    path = PATH_TO_GAMES + "/train/*.ulx"
    games = glob.glob(path)

     generate_data(games,5154,5)