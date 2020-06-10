import numpy as np
import torch
import random
import uuid
import os
import re
import time
import string
from collections import Counter
from os.path import join as pjoin
from functools import lru_cache

import textworld
from textworld.logic import State, Rule, Proposition, Variable
missing_words = set()


##############################
# KG stuff
##############################
# relations
two_args_relations = ["in", "on", "at", "west_of", "east_of", "north_of", "south_of", "part_of", "needs"]
one_arg_state_relations = ["chopped", "roasted", "diced", "burned", "open", "fried", "grilled", "consumed", "closed", "sliced", "uncut", "raw"]
ignore_relations = ["cuttable", "edible", "drinkable", "sharp", "inedible", "cut", "cooked", "cookable", "needs_cooking"]
opposite_relations = {"west_of": "east_of",
                      "east_of": "west_of",
                      "south_of": "north_of",
                      "north_of": "south_of"}
equivalent_entities = {"inventory": "player",
                       "recipe": "cookbook"}
FOOD_FACTS = ["sliced", "diced", "chopped", "cut", "uncut", "cooked", "burned",
              "grilled", "fried", "roasted", "raw", "edible", "inedible"]


def process_equivalent_entities_in_triplet(triplet):
    # ["cookbook", "inventory", "in"]
    for i in range(len(triplet)):
        if triplet[i] in equivalent_entities:
            triplet[i] = equivalent_entities[triplet[i]]
    return triplet


def process_equivalent_entities_in_command(command):
    # "add , knife , inventory , in"
    words = command.split(" , ")
    words = [item.strip() for item in words]
    for i in range(len(words)):
        if words[i] in equivalent_entities:
            words[i] = equivalent_entities[words[i]]
    return " , ".join(words)


def process_exits_in_triplet(triplet):
    # ["exit", "kitchen", "backyard", "south_of"]
    if triplet[0] == "exit":
        print(triplet)
        return [triplet[0], triplet[1], triplet[3]]
    else:
        return triplet


def process_exits_in_command(command):
    # "add , exit , kitchen , corridor , east_of"
    words = command.split(" , ")
    words = [item.strip() for item in words]
    if words[1] == "exit":
        return " , ".join([words[0], words[1], words[2], words[4]])
    else:
        return command


def process_burning_triplets(list_of_triplets):
    burned_stuff = []
    for t in list_of_triplets:
        if "burned" in t:
            burned_stuff.append(t[0])
    res = []
    for t in list_of_triplets:
        if t[0] in burned_stuff and t[1] in ["grilled", "fried", "roasted"]:
            continue
        res.append(t)
    return res


def process_burning_commands(list_of_commands, list_of_triplets):
    cook = set(["grilled", "fried", "roasted"])
    burned_stuff = []
    for c in list_of_commands:
        if "burned" in c:
            burned_stuff.append(c.split(",")[1].strip())
    res = []
    for bs in burned_stuff:
        for t in list_of_triplets:
            if bs not in t:
                continue
            intersection = set(t) & cook
            if len(intersection) == 0:
                continue
            res.append("delete , " + bs + " , " + list(intersection)[0] + " , is")
            break
    return list_of_commands +  res


def process_direction_triplets(list_of_triplets):
    res = []
    for t in list_of_triplets:
        res.append(t)
        if t[0] == "exit" or t[1] == "exit":
            continue
        if "north_of" in t:
            res.append([t[1], t[0], "south_of"])
        elif "south_of" in t:
            res.append([t[1], t[0], "north_of"])
        elif "east_of" in t:
            res.append([t[1], t[0], "west_of"])
        elif "west_of" in t:
            res.append([t[1], t[0], "east_of"])
    return res


@lru_cache()
def _rules_predicates_scope():
    rules = [
        Rule.parse("query :: at(P, r) -> at(P, r)"),
        Rule.parse("query :: at(P, r) & at(o, r) -> at(o, r)"),
        Rule.parse("query :: at(P, r) & at(d, r) -> at(d, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) -> at(s, r)"),
        Rule.parse("query :: at(P, r) & at(c, r) -> at(c, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) & on(o, s) -> on(o, s)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) -> open(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & closed(c) -> closed(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(o, c) -> in(o, c)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & open(d) -> open(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & closed(d) -> closed(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    rules += [Rule.parse("query :: at(P, r) & at(f, r) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(s, r) & on(f, s) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(f, c) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_predicates_recipe():
    rules = [
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) -> part_of(f, RECIPE)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & roasted(ingredient) -> needs_roasted(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & grilled(ingredient) -> needs_grilled(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & fried(ingredient) -> needs_fried(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & sliced(ingredient) -> needs_sliced(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & chopped(ingredient) -> needs_chopped(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & diced(ingredient) -> needs_diced(f)"),
    ]
    return rules


@lru_cache()
def _rules_exits():
    rules = [
        Rule.parse("query :: at(P, r) & north_of(r', r) -> north_of(r', r)"),
        Rule.parse("query :: at(P, r) & west_of(r', r) -> west_of(r', r)"),
        Rule.parse("query :: at(P, r) & south_of(r', r) -> south_of(r', r)"),
        Rule.parse("query :: at(P, r) & east_of(r', r) -> east_of(r', r)"),
    ]
    return rules


@lru_cache()
def _rules_predicates_inv():
    rules = [
        Rule.parse("query :: in(o, I) -> in(o, I)"),
    ]
    rules += [Rule.parse("query :: in(f, I) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_to_convert_link_predicates():
    rules = [
        Rule.parse("query :: link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    return rules


def find_predicates_in_scope(state):
    actions = state.all_applicable_actions(_rules_predicates_scope())
    return [action.postconditions[0] for action in actions]


def find_exits_in_scope(state):
    actions = state.all_applicable_actions(_rules_exits())

    def _convert_to_exit_fact(proposition):
        return Proposition(proposition.name,
                           [Variable("exit", "LOCATION"),
                            proposition.arguments[1],
                            proposition.arguments[0]])

    return [_convert_to_exit_fact(action.postconditions[0]) for action in actions]


def convert_link_predicates(state):
    actions = state.all_applicable_actions(_rules_to_convert_link_predicates())
    for action in list(actions):
        state.apply(action)
    return state


def find_predicates_in_inventory(state):
    actions = state.all_applicable_actions(_rules_predicates_inv())
    return [action.postconditions[0] for action in actions]


def find_predicates_in_recipe(state):
    actions = state.all_applicable_actions(_rules_predicates_recipe())

    def _convert_to_needs_relation(proposition):
        if not proposition.name.startswith("needs_"):
            return proposition

        return Proposition("needs",
                           [proposition.arguments[0],
                            Variable(proposition.name.split("needs_")[-1], "STATE")])

    return [_convert_to_needs_relation(action.postconditions[0]) for action in actions]


def process_facts(prev_facts, info_game, info_facts, info_last_action, cmd):
    kb = info_game.kb
    if prev_facts is None or cmd == "restart":
        facts = set()
    else:
        if cmd == "inventory":  # Bypassing TextWorld's action detection.
            facts = set(find_predicates_in_inventory(State(kb.logic, info_facts)))
            return prev_facts | facts

        elif info_last_action is None :
            return prev_facts  # Invalid action, nothing has changed.

        elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
            facts = set(find_predicates_in_recipe(State(kb.logic, info_facts)))
            return prev_facts | facts

        state = State(kb.logic, prev_facts | set(info_last_action.preconditions))
        success = state.apply(info_last_action)
        assert success
        facts = set(state.facts)

    # Always add facts in sight.
    facts |= set(find_predicates_in_scope(State(kb.logic, info_facts)))
    facts |= set(find_exits_in_scope(State(kb.logic, info_facts)))

    return facts

def process_local_facts(info_game, info_facts):
    state = State(info_game.kb.logic, info_facts)
    scope_facts = set(find_predicates_in_scope(state))
    exit_facts = set(find_exits_in_scope(state))
    inventory_facts = set(find_predicates_in_inventory(state))
    recipe_facts = set(find_predicates_in_recipe(state))
    return scope_facts | exit_facts | inventory_facts | recipe_facts 

def process_fully_obs_facts(info_game, facts):
    state = State(info_game.kb.logic, facts)
    state = convert_link_predicates(state)
    inventory_facts = set(find_predicates_in_inventory(state))
    recipe_facts = set(find_predicates_in_recipe(state))
    return set(state.facts) | inventory_facts | recipe_facts


def process_local_obs_facts(info_game, info_facts, info_last_action, cmd):
    def _get_state():
        return State(info_game.kb.logic, info_facts)

    if cmd == "inventory":  # Bypassing TextWorld's action detection.
        return set(find_predicates_in_inventory(_get_state()))

    elif (info_last_action and info_last_action.name.startswith("go")) or cmd in ["restart", "look"]:
        # Facts in sight.
        state = _get_state()
        facts = set(find_predicates_in_scope(state))
        facts |= set(find_exits_in_scope(state))
        return facts

    elif info_last_action is None:
        return set()  # Invalid action, no facts.

    elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
        return set(find_predicates_in_recipe(_get_state()))

    return info_last_action.postconditions



def serialize_facts(facts):
    PREDICATES_TO_DISCARD = {"ingredient_1", "ingredient_2", "ingredient_3", "ingredient_4", "ingredient_5",
                             "out", "free", "used", "cooking_location", "link"}
    CONSTANT_NAMES = {"P": "player", "I": "player", "ingredient": None, "slot": None, "RECIPE": "cookbook"}
    # e.g. [("wooden door", "backyard", "in"), ...]
    serialized = [[arg.name if arg.name and arg.type not in CONSTANT_NAMES else CONSTANT_NAMES[arg.type] for arg in fact.arguments] + [fact.name]
                    for fact in sorted(facts) if fact.name not in PREDICATES_TO_DISCARD]
    #print(facts,serialized,[fact for fact in serialized if None not in fact])
    return filter_triplets([fact for fact in serialized if None not in fact])


def filter_triplets(triplets):
    tp = []
    for item in triplets:
        item = process_equivalent_entities_in_triplet(item)
        item = process_exits_in_triplet(item)
        
        if item[-1] in (two_args_relations +  one_arg_state_relations):
            tp.append([it.lower() for it in item])
        else:
            if item[-1] not in ignore_relations:
                print("Warning..., %s not in known relations..." % (item[-1]))
    for i in range(len(tp)):
        if tp[i][-1] in one_arg_state_relations:
            tp[i].append("is")
    tp = process_burning_triplets(tp)
    tp = process_direction_triplets(tp)
    return tp