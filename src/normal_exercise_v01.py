import src.exercise as exercise

import simplenlg as nlg
from simplenlg import LexicalCategory as lexcat
from simplenlg import Feature as f
from simplenlg import Tense as t
import random
from copy import deepcopy

def get_mean():
    return random.choice(['average', 'mean', 'expected value'])

def get_std():
    return random.choice(['standard deviation', 'variance'])

def get_prob():
    return random.choice(['probability','chance'])

def get_approx():
    return random.choice(['', '~', 'approximately '])

def pre_post(c,mod):
    if random.getrandbits(1):
        c.setPreModifier(mod)
    else:
        c.setPostModifier(mod)
    return c

class NormalExercise(exercise.Exercise):
    def __init__(self, config):

        lexicon = nlg.Lexicon.getDefaultLexicon()
        factory = nlg.NLGFactory(lexicon)
        realiser = nlg.Realiser(lexicon)

        def c_to_s(c):
            return str(realiser.realise(factory.createSentence(c)))

        super().__init__(config=config)
        self.config = config

        ### STATEMENTS ###
        self.statements = []

        # Normal distribution assumption
        subj = deepcopy(self.config['SUBJ'])
        subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

        rand2 = random.choice(range(6))
        options2 = {
            0:"Empirical data has shown that",
            1:"Experience has shown that",
            2:"It is known that",
            3:"It's generally known that",
            4:"It's a generally accepted fact that",
            5:"It's generally established that"
        }
        rand = random.choice(range(3))
        options = {
            0:f"{options2[rand2]} the distribution of {realiser.realise(subj)} can be assumed to be normal.",
            1:f"{options2[rand2]} {realiser.realise(subj)} can be assumed to follow a normal distribution.",
            2:f"{options2[rand2]} {realiser.realise(subj)} tend to follow a normal distribution.",
        }
        self.statements.append(options[rand])

        # Mean assumption
        mu = (random.randrange(0, 30, 2)-15)/100 * int(self.config['VALRANGE'])  + int(self.config['VALRANGE'])
        if mu>25:
            mu=int(mu)
        else:
            mu=round(mu,1)
        subj = deepcopy(self.config['SUBJ'])

        rand = random.choice(range(6))
        options = {
            0:f"The {get_mean()} {realiser.realise(subj)} is {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
            1:f"The {get_mean()} {realiser.realise(subj)} is assumed to be {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
            2:f"The {get_mean()} {realiser.realise(subj)} is known to be {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
            3:f"The {get_mean()} {realiser.realise(subj)} is generally known to be {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
            4:f"Empirical data has also shown that the {get_mean()} {realiser.realise(subj)} is {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
            5:f"Experts have shown that the {get_mean()} {realiser.realise(subj)} is {get_approx()}{mu} {realiser.realise(self.config['UNIT'])}.",
        }
        self.statements.append(options[rand])

        # StdDev assumption
        std = random.randrange(3, 15, 1)/100 * int(self.config['VALRANGE'])
        if std>25:
            std=int(std)
        else:
            if std==round(std,1)>0:
                std=round(std,1)
            else:
                std=round(std*3,1)
        subj = deepcopy(self.config['SUBJ'])

        rand = random.choice(range(6))
        options = {
            0:f"The {get_std()} in {realiser.realise(subj)} is {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
            1:f"The {get_std()} in {realiser.realise(subj)} is assumed to be {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
            2:f"The {get_std()} in {realiser.realise(subj)} is known to be {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
            3:f"The {get_std()} in {realiser.realise(subj)} is generally known to be {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
            4:f"Empirical data has also shown that the {get_std()} in {realiser.realise(subj)} is {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
            5:f"Experts have shown that the {get_std()} in {realiser.realise(subj)} is {get_approx()}{std} {realiser.realise(self.config['UNIT'])}.",
        }
        self.statements.append(options[rand])

        ### QUESTIONS ###
        self.questions = []
        for iter in range(3):
            # ----------------------------------------------------------------------------------
            # What is the chance of a measuring a height below X?
            for subiter in range(2):
                X = mu + random.choice([-1,1]) * random.randrange(0, 300, 10)/100 * std
                if X>25:
                    X=int(X)
                else:
                    X=round(X,1)

                subj = deepcopy(self.config['SUBJ'])

                rand2 = random.choice(range(5))
                options2 = {
                    0:(f"How big is","?"),
                    1:(f"What is","?"),
                    2:(f"Determine","."),
                    3:(f"Calculate","."),
                    4:(f"Compute","."),
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"of at most ",
                    1:f"of at least ",
                    2:"above ",
                    3:"below "
                }
                rand = random.choice(range(3))
                options = {
                    0:f"{options2[rand2][0]} the {get_prob()} of measuring a {realiser.realise(subj)} {options3[rand3]}{X} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    1:f"{options2[rand2][0]} the {get_prob()} of a {realiser.realise(subj)} being {options3[rand3]}{X} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    2:f"{options2[rand2][0]} the {get_prob()} that a {realiser.realise(subj)} is {options3[rand3]}{X} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                }

                self.questions.append(('easy',options[rand]))
            # ----------------------------------------------------------------------------------
            # What is the 90th percentile of the distribution of heights?
            for subiter in range(2):
                if bool(random.getrandbits(1)):
                    X = random.randrange(5, 25, 5)
                else:
                    X = random.randrange(75, 95, 5)

                subj = deepcopy(self.config['SUBJ'])
                subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                rand2 = random.choice(range(4))
                options2 = {
                    0:(f"What is","?"),
                    1:(f"Determine","."),
                    2:(f"Calculate","."),
                    3:(f"Compute","."),
                }
                rand = random.choice(range(4))
                options = {
                    0:f"{options2[rand2][0]} the {X}th percentile of the distribution of {realiser.realise(subj)}{options2[rand2][1]}",
                    1:f"Given the assumptions about the distribution of {realiser.realise(subj)}, calculate its {X}th percentile.",
                    2:f"Given the assumptions about the distribution of {realiser.realise(subj)}, compute its {X}th percentile.",
                    3:f"Given the assumptions about the distribution of {realiser.realise(subj)}, determine its {X}th percentile.",
                }

                self.questions.append(('easy',options[rand]))
            # ----------------------------------------------------------------------------------
            # Given X measurements, what is the chance that exactly Y are below Z?
            for subiter in range(2):
                X = random.randrange(10, 25)
                Y = X - random.randrange(1, X-1)
                Z = mu + random.choice([-1,1]) * random.randrange(0, 150, 10)/100 * std

                subj = deepcopy(self.config['SUBJ'])
                subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                rand2 = random.choice(range(4))
                options2 = {
                    0:(f"What is","?"),
                    1:(f"Determine","."),
                    2:(f"Calculate","."),
                    3:(f"Compute","."),
                }
                rand3 = random.choice(range(3))
                options3 = {
                    0:"exactly",
                    1:"precisely",
                    2:"",
                }
                rand4 = random.choice(range(4))
                options4 = {
                    0:"lower than ",
                    1:"higher than ",
                    2:"above ",
                    3:"below "
                }
                rand = random.choice(range(3))
                options = {
                    0:f"Given that {X} {realiser.realise(subj)} are measured, {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options4[rand4]}{Z} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    1:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options4[rand4]}{Z} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    2:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the expected value of measurements {options4[rand4]}{Z} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                }

                self.questions.append(('medium',options[rand]))
            # ----------------------------------------------------------------------------------
            # Given X measurements, what is the chance that exactly Y are below Z?
            for subiter in range(2):
                X = random.randrange(10, 25)
                Y = X - random.randrange(1, X-1)

                Z_high=1
                Z_low=1
                i = 0
                while Z_high/Z_low-1<0.1:
                    Z_high = mu + random.choice([-1,1]) * random.randrange(0, 150, 10)/100 * std
                    Z_low = mu + random.choice([-1,1]) * random.randrange(0, 150, 10)/100 * std
                    if Z_high>25:
                        Z_high=int(Z_high)
                    else:
                        Z_high=round(Z_high,1)
                    if Z_low>25:
                        Z_low=int(Z_low)
                    else:
                        Z_low=round(Z_low,1)
                    if Z_low>Z_high:
                        Z_high,Z_low=Z_low,Z_high
                    if i > 10:
                        break
                    i += 1

                subj = deepcopy(self.config['SUBJ'])
                subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                rand2 = random.choice(range(4))
                options2 = {
                    0:(f"What is","?"),
                    1:(f"Determine","."),
                    2:(f"Calculate","."),
                    3:(f"Compute","."),
                }
                rand3 = random.choice(range(3))
                options3 = {
                    0:"exactly ",
                    1:"precisely ",
                    2:"",
                }
                rand4 = random.choice(range(2))
                options4 = {
                    0:"lower than ",
                    1:"below "
                }
                rand5 = random.choice(range(2))
                options5 = {
                    0:"higher than ",
                    1:"above ",
                }
                rand = random.choice(range(6))
                options = {
                    0:f"Given that {X} {realiser.realise(subj)} are measured, {options2[rand2][0]} the {get_prob()} that {options3[rand3]}{Y} measurements are {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])} and {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    1:f"Given that {X} {realiser.realise(subj)} are measured, {options2[rand2][0]} the {get_prob()} that {options3[rand3]}{Y} measurements are {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])} or {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    2:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the {get_prob()} that {options3[rand3]}{Y} measurements are {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])} and {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    3:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the {get_prob()} that {options3[rand3]}{Y} measurements are {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])} or {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    4:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the expected value of measurements {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])} and {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    5:f"Assume that {X} {realiser.realise(subj)} have been measured. {options2[rand2][0]} the expected value of measurements {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])} or {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                }

                self.questions.append(('medium',options[rand]))
            # ----------------------------------------------------------------------------------
            # Given X measurements, what is the chance that at least Y are below Z or above A?
            for subiter in range(2):
                X = random.randrange(10, 25)
                Y = X - random.randrange(1, X-1)

                i = 0
                while Z_high/Z_low-1<0.1:
                    Z_high = mu + random.choice([-1,1]) * random.randrange(0, 150, 10)/100 * std
                    Z_low = mu + random.choice([-1,1]) * random.randrange(0, 150, 10)/100 * std
                    if Z_high>25:
                        Z_high=int(Z_high)
                    else:
                        Z_high=round(Z_high,1)
                    if Z_low>25:
                        Z_low=int(Z_low)
                    else:
                        Z_low=round(Z_low,1)
                    if Z_low>Z_high:
                        Z_high,Z_low=Z_low,Z_high
                    if i > 10:
                        break
                    i += 1

                subj = deepcopy(self.config['SUBJ'])

                rand2 = random.choice(range(4))
                options2 = {
                    0:(f"What is","?"),
                    1:(f"Determine","."),
                    2:(f"Calculate","."),
                    3:(f"Compute","."),
                }
                rand3 = random.choice(range(6))
                options3 = {
                    0:"at least",
                    1:"at most",
                    2:"at maximum",
                    3:"not less than",
                    4:"not more than",
                    5:"a minimum of"
                }
                rand4 = random.choice(range(2))
                options4 = {
                    0:"lower than ",
                    1:"below "
                }
                rand5 = random.choice(range(2))
                options5 = {
                    0:"higher than ",
                    1:"above ",
                }
                rand = random.choice(range(4))
                options = {
                    0:f"Given {X} measurements of {realiser.realise(subj)}, {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])} and {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    1:f"Given {X} measurements of {realiser.realise(subj)}, {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])} or {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    2:f"Assume {X} measurements of {realiser.realise(subj)} have been collected. {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])} and {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                    3:f"Assume {X} measurements of {realiser.realise(subj)} have been collected. {options2[rand2][0]} the {get_prob()} that {options3[rand3]} {Y} measurements are {options5[rand5]}{Z_high} {realiser.realise(self.config['UNIT'])} or {options4[rand4]}{Z_low} {realiser.realise(self.config['UNIT'])}{options2[rand2][1]}",
                }

                self.questions.append(('hard',options[rand]))
            # ----------------------------------------------------------------------------------
            # Outliers are defined as measurements whose absolute distance to the mean are more than X std. How many outliers have to be measured in a row for that event to have a probability of less than 1%?
            for subiter in range(1):
                X = random.randrange(10, 25)
                Y = random.randrange(3,4,1)/2
                Z = random.randrange(1,5)/(10**random.randrange(1,3,1))

                subj = deepcopy(self.config['SUBJ'])

                rand2 = random.choice(range(4))
                options2 = {
                    0:(f"What is the amount of outliers that","?"),
                    1:(f"Determine the amount of outliers that","."),
                    2:(f"Calculate the number of outliers that","."),
                    3:(f"How many outliers","?"),
                }
                rand3 = random.choice(range(3))
                options3 = {
                    0:"in a row",
                    1:"in direct succession",
                    2:f"in a set of {X} measurements"
                }
                rand4 = random.choice(range(3))
                options4 = {
                    0:"less than",
                    1:"below",
                    2:"at most"
                }
                rand = random.choice(range(3))
                options = {
                    0:f"Outliers are defined as measurements whose absolute distance to the mean is more than {Y} StdDev. {options2[rand2][0]} have to be detected {options3[rand3]} for the event to have a {get_prob()} of {options4[rand4]} {Z}%{options2[rand2][1]}",
                    1:f"Outliers can be defined as measurements whose absolute distance to the mean is more than {Y} StdDev. {options2[rand2][0]} have to be detected {options3[rand3]} for the event to have a {get_prob()} of {options4[rand4]} {Z}%{options2[rand2][1]}",
                    2:f"Experts define Outliers to be measurements whose absolute distance to the mean is more than {Y} StdDev. {options2[rand2][0]} have to be detected {options3[rand3]} for the event to have a {get_prob()} of {options4[rand4]} {Z}%{options2[rand2][1]}",
                }

                self.questions.append(('hard',options[rand]))
