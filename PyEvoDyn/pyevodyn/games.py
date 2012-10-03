'''
Created on Oct 2, 2012

@author: garcia

Famous games, symbolic and numeric
'''

import numpy as np

#TODO: Document
#TODO: symbolic versions

def prisoners_dilemma(reward=3.0,sucker=0.0,temptation=4.0,punishment=1.0):
    return np.array([[reward,sucker],[temptation,punishment]])

def prisoners_dilemma_equal_gains(benefit=2,cost=1):
    return prisoners_dilemma(reward=benefit-cost,sucker=-cost,temptation=benefit,punishment=0)


def rock_paper_sccisors(win=1.0,lose=-1.0, tie=0.0):
    return np.array([[tie,lose, win],[win,tie, lose],[lose,win, tie]])

def hawk_dove_game(resource_value=2.0, cost_of_fighting=1.0):
    return np.array([[(resource_value-cost_of_fighting)/2.0,resource_value],[0,resource_value]])

def stag_hunt(stag=2.0, hare=1.0):
    return np.array([[stag,0.0],[hare,hare]])


def allc_tft_alld(reward=3.0,sucker=0.0,temptation=4.0,punishment=1.0,continuation_probability=0.95):
    return np.array([[reward,reward,sucker],[reward,reward, sucker*(1.0-continuation_probability)+punishment*continuation_probability],[temptation,temptation*(1.0-continuation_probability)+punishment*continuation_probability, punishment]])

def allc_tft_alld_equal_gains(benefit=2.0, cost=1.0,continuation_probability=0.95):
    reward=benefit-cost
    sucker=-cost
    temptation=benefit
    punishment=0.0
    return np.array([[reward,reward,sucker],[reward,reward, sucker*(1.0-continuation_probability)+punishment*continuation_probability],[temptation,temptation*(1.0-continuation_probability)+punishment*continuation_probability, punishment]])