
Code for Master's Project 
Report: [Exploring Generalization in Text-based Reinforcement Learning ](https://bryonkucharski.github.io/files/ms-project-final-report.pdf)

# Goal
Goal was to test a reinforcement learning agent's ability to generalize to unseen text-based games

# Model Architecture
![architecture](https://github.com/bryonkucharski/text-based-rl-generalization/blob/master/images/architecture.jpg)

# Example Input
Input Type | Example Text |
---   | ---   | 
Description     | -= Kitchen =-If you're wondering why everything seems so normal all of a sudden, it's because you've  &  just shown up in the kitchen. You can see a closed fridge, which looks conventional, right there by you. You see a closed oven right there by you. Oh, great. Here's a table. Unfortunately, there isn't a thing on it. Hm.  Oh well You scan the room, seeing a counter. The counter is vast.  On the counter you can make out a cookbook and a knife. You make out a stove. Looks like someone's already been here and taken everything off it, though. Sometimes, just sometimes, TextWorld can just be the worst. |
Inventory |   You are carrying:  a banana |
Recipe |      Recipe #1 - Gather all following ingredients and follow the directions to prepare this tasty meal. Ingredients: banana Directions: chop the banana fry the banana prepare meal |
Previous Action | take banana from counter |
Feedback | You take the banana from the counter. Your score has just gone up by one point |

# Results
Plots across four Levels of difficulties (left to right)

Normalized score is y axis

## Training One Game at a time
![single_game](https://github.com/bryonkucharski/text-based-rl-generalization/blob/master/images/single_game_plots.jpg)

## Evaluation Results across 20 evaluation games
![Evaluation](https://github.com/bryonkucharski/text-based-rl-generalization/blob/master/images/multi_game_plots.jpg)
