import src.exercise as exercise

import simplenlg as nlg
from simplenlg import LexicalCategory as lexcat
from simplenlg import Feature as f
from simplenlg import Tense as t
import random
from copy import deepcopy

def get_prob():
    return random.choice(['probability','chance'])

def pre_post(c,mod):
    if random.getrandbits(1):
        c.setPreModifier(mod)
    else:
        c.setPostModifier(mod)
    return c

class BinomialExercise(exercise.Exercise):
    def __init__(self, config):
        # Two attributes: {'CONJ': 'hit', 'ACOMP': 'missed', 'SUBJ': 'shot', 'VERB': 'is', 'OBJ': None, 'UNIT': None, 'VALRANGE': None}

        lexicon = nlg.Lexicon.getDefaultLexicon()
        factory = nlg.NLGFactory(lexicon)
        realiser = nlg.Realiser(lexicon)

        def c_to_s(c):
            return str(realiser.realise(factory.createSentence(c)))

        super().__init__(config=config)
        self.config = config

        ### STATEMENTS ###
        self.statements = []

        if bool(random.getrandbits(1)):
            # --- 3
            # A traffic light is red with a probability of 95%.
            p = random.randrange(10, 100, 5)

            subj = deepcopy(self.config['SUBJ'])
            if bool(random.getrandbits(1)):
                subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)
            else:
                subj.setDeterminer("a")

            verb = factory.createVerbPhrase(self.config['VERB'])

            if bool(random.getrandbits(1)):
                obj = factory.createNounPhrase(self.config['ACOMP'])
            else:
                obj = factory.createNounPhrase(self.config['CONJ'])

            clause = factory.createClause(subj, verb, obj)
            clause.addPostModifier(f"with a {get_prob()} of {p}%")

            self.statements.append(c_to_s(clause))
        else:
            p = random.randrange(10, 100, 5)
            subj = deepcopy(self.config['SUBJ'])
            
            if bool(random.getrandbits(1)):
                s = f"The {get_prob()} of a {realiser.realise(subj)} being {self.config['CONJ']} is {p}%."
            else:
                s = f"The {get_prob()} of a {realiser.realise(subj)} being {self.config['ACOMP']} is {p}%."
            self.statements.append(s)

        if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
            # --- 4
            # A traffic light shows either red or green
            subj = deepcopy(self.config['SUBJ'])
            subj.setDeterminer("a")
            if bool(random.getrandbits(1)):
                subj.setPlural = True

            verb = factory.createVerbPhrase(self.config['VERB'])
            verb.setPreModifier('only')
            
            if self.config.get('ACOMP', False):
                obj = factory.createNounPhrase(self.config['ACOMP'])
            else:
                obj = factory.createNounPhrase(self.config['OPRD'])
            obj.setPreModifier('either')
            obj_2 = factory.createNounPhrase(self.config['CONJ'])
            obj_2.setPreModifier('or')
            obj.setPostModifier(obj_2)

            clause = factory.createClause(subj, verb, obj)

            self.statements.append(c_to_s(clause))

        if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
            # --- 4
            # Every Traffic lights chances of showing either red or green are independent.
            subj = deepcopy(self.config['SUBJ'])
            subj.setDeterminer("every")
            subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

            if bool(random.getrandbits(1)):
                chance = factory.createNounPhrase('chance')
                if bool(random.getrandbits(1)):
                    chance.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)
            else:
                chance = factory.createNounPhrase('probability')

            verb = factory.createVerbPhrase(self.config['VERB'])
            verb.setPreModifier('of')
            verb.setFeature(f.FORM, nlg.Form.GERUND)

            if self.config.get('ACOMP', False):
                obj = factory.createNounPhrase(self.config['ACOMP'])
            else:
                obj = factory.createNounPhrase(self.config['OPRD'])
            obj.setPreModifier('either')
            obj_2 = factory.createNounPhrase(self.config['CONJ'])
            obj_2.setPreModifier('or')
            obj.setPostModifier(obj_2)
            verb.setPostModifier(obj)

            chance.setPostModifier(verb)

            verb_2 = factory.createVerbPhrase('be')
            verb_2.setPostModifier('independent')
            verb_2.setPlural(True)

            clause = factory.createClause()
            clause.setSubject(subj)
            clause.setVerb(chance)
            clause.setPostModifier(verb_2)

            self.statements.append(c_to_s(clause))


        ### QUESTIONS ###
        self.questions = []
        for iter in range(1):
            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 1
                # Calculate the probability that 8 out of 12 Traffic lights are green.
                Y = random.randrange(6, 13)
                X = Y - random.randrange(1, 5)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                else:
                    clause.setObject(self.config['CONJ'])

                rand = random.choice(range(4))
                options = {
                    0:f"Calculate the {get_prob()} that {X} out of {Y}",
                    1:f"Determine the {get_prob()} that {X} out of {Y}",
                    2:f"State the {get_prob()} that {X} out of {Y}",
                    3:f"Compute the {get_prob()} that {X} out of {Y}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('easy',c_to_s(clause)))
            
            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 1
                # What is the probability that 8 out of 12 Traffic lights are green?
                Y = random.randrange(6, 13)
                X = Y - random.randrange(1, 5)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                else:
                    clause.setObject(self.config['CONJ'])

                rand = random.choice(range(2))
                options = {
                    0:f"What is the {get_prob()} that {X} out of {Y}",
                    1:f"How big is the {get_prob()} that {X} out of {Y}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('easy',c_to_s(clause)[:-1]+'?'))

    # -----------------------------------------------------------------------------

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # What is the chance to hit exactly 7 times in a row?
                X = random.randrange(4, 8)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(2))
                rand2 = random.choice(range(7))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:f"at most",
                    3:f"exactly",
                    4:f"",
                    5:"a minimum of",
                    6:"a maximum of"
                }
                options = {
                    0:f"What is the {get_prob()} that {options2[rand2]} {X} ",
                    1:f"How big is the {get_prob()} that {options2[rand2]} {X} ",
                }
                clause.setFrontModifier(options[rand])

                rand = random.choice(range(4))
                options = {
                    0:f"in a row",
                    1:f"directly after one another",
                    2:f"in direct succession",
                    3:f"consecutively",
                }
                clause.setPostModifier(options[rand])

                self.questions.append(('medium',c_to_s(clause)[:-1]+'?'))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Calculate the probability that at least 5 shots are missed in a row
                X = random.randrange(4, 8)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(2))
                rand2 = random.choice(range(7))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:f"at most",
                    3:f"exactly",
                    4:f"",
                    5:"a minimum of",
                    6:"a maximum of"
                }
                options = {
                    0:f"Calculate the {get_prob()} that {options2[rand2]} {X} ",
                    1:f"Compute the {get_prob()} that {options2[rand2]} {X} ",
                    2:f"Determine the {get_prob()} that {options2[rand2]} {X} ",
                }
                clause.setFrontModifier(options[rand])

                rand = random.choice(range(4))
                options = {
                    0:f"in a row",
                    1:f"directly after one another",
                    2:f"in direct succession",
                    3:f"consecutively",
                }
                clause.setPostModifier(options[rand])

                self.questions.append(('medium',c_to_s(clause)))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Calculate the probability that at least 5 and at most 7 shots are missed in a row
                X = random.randrange(4, 8)
                Y = X + random.randrange(2,4)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(3))
                rand2 = random.choice(range(4))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than"
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"Calculate the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    1:f"Compute the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    2:f"Determine the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                }
                clause.setFrontModifier(options[rand])

                rand = random.choice(range(4))
                options = {
                    0:f"in a row",
                    1:f"directly after one another",
                    2:f"in direct succession",
                    3:f"consecutively",
                }
                clause.setPostModifier(options[rand])

                self.questions.append(('hard',c_to_s(clause)))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # What is the probability that at least 5 and at most 7 shots are missed in a row?
                X = random.randrange(4, 8)
                Y = X + random.randrange(2,4)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(2))
                rand2 = random.choice(range(4))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than"
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"How big is the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    1:f"What is the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                }
                clause.setFrontModifier(options[rand])

                rand = random.choice(range(4))
                options = {
                    0:f"in a row",
                    1:f"directly after one another",
                    2:f"in direct succession",
                    3:f"consecutively",
                }
                clause.setPostModifier(options[rand])

                self.questions.append(('hard',c_to_s(clause)[:-1]+'?'))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Given X shots, what is the chance that between 3 and 6 incl. 
                Z = random.randrange(12)
                X = random.randrange(4, 8)
                Y = X + random.randrange(2,4)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(4))
                rand2 = random.choice(range(4))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than"
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"Given {Z} {realiser.realise(subj)}, how big is the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    1:f"Given {Z} {realiser.realise(subj)}, what is the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    2:f"Given {Z} {realiser.realise(subj)}, how big is the {get_prob()} that {options2[rand2]} {Y} or {options3[rand3]} {X}",
                    3:f"Given {Z} {realiser.realise(subj)}, what is the {get_prob()} that {options2[rand2]} {Y} or {options3[rand3]} {X}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('medium',c_to_s(clause)[:-1]+'?'))
            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Given X shots, what is the chance that between 3 and 6 incl?
                Z = random.randrange(12, 14)
                X = random.randrange(4, 8)
                Y = X + random.randrange(2,4)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(6))
                rand2 = random.choice(range(4))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than"
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"Given {Z} {realiser.realise(subj)} calculate the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    1:f"Given {Z} {realiser.realise(subj)} compute the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    2:f"Given {Z} {realiser.realise(subj)} determine the {get_prob()} that {options2[rand2]} {X} and {options3[rand3]} {Y}",
                    3:f"Given {Z} {realiser.realise(subj)} calculate the {get_prob()} that {options2[rand2]} {Y} or {options3[rand3]} {X}",
                    4:f"Given {Z} {realiser.realise(subj)} compute the {get_prob()} that {options2[rand2]} {Y} or {options3[rand3]} {X}",
                    5:f"Given {Z} {realiser.realise(subj)} determine the {get_prob()} that {options2[rand2]} {Y} or {options3[rand3]} {X}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('medium',c_to_s(clause)))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Given X shots, what is the chance that between 3 and 6 incl?
                Z = random.randrange(7,11)
                X = random.randrange(3, 7)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(3))
                rand2 = random.choice(range(4))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than"
                }
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"Given {Z} {realiser.realise(subj)} calculate the {get_prob()} that {options2[rand2]} {X}",
                    1:f"Given {Z} {realiser.realise(subj)} compute the {get_prob()} that {options2[rand2]} {X}",
                    2:f"Given {Z} {realiser.realise(subj)} determine the {get_prob()} that {options2[rand2]} {X}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('easy',c_to_s(clause)))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Given X shots, what is the chance that at least 5 are hit
                Z = random.randrange(7,11)

                subj = deepcopy(self.config['SUBJ'])
                subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(2))
                options = {
                    0:f"Given {Z} {realiser.realise(subj)}, what is the expected amount of {alt} {realiser.realise(subj)}?",
                    1:f"Given {Z} {realiser.realise(subj)}, how big is the expected number of {alt} {realiser.realise(subj)}?",
                }
                self.questions.append(('easy',options[rand]))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # Given X shots, what is the chance that at least 5 are hit
                Z = random.randrange(7,11)
                X = random.randrange(3, 7)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')

                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(2))
                rand2 = random.choice(range(8))
                options2 = {
                    0:f"at least",
                    1:f"at minimum",
                    2:"a minimum of",
                    3:"not less than",
                    4:f"at most",
                    5:f"at maximum",
                    6:"a maximum of",
                    7:"not more than"
                }
                options = {
                    0:f"Given {Z} {realiser.realise(subj)}, what is the {get_prob()} that {options2[rand2]} {X}",
                    1:f"Given {Z} {realiser.realise(subj)}, how big is the {get_prob()} that {options2[rand2]} {X}",
                }
                clause.setFrontModifier(options[rand])

                self.questions.append(('easy',c_to_s(clause)[:-1]+'?'))

            if self.config.get('ACOMP', False) or self.config.get('OPRD', False):
                # --- 2
                # How many shots must be hit in a row for that event to have a probability of less than X 
                p = random.randrange(5, 20, 1)

                subj = deepcopy(self.config['SUBJ'])
                if X > 1:
                    subj.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

                verb = factory.createVerbPhrase('is')


                clause = factory.createClause()
                clause.setSubject(subj)
                clause.setVerb(verb)
                if bool(random.getrandbits(1)):
                    if self.config.get('ACOMP', False):
                        clause.setObject(self.config['ACOMP'])
                        alt = self.config['CONJ']
                else:
                    clause.setObject(self.config['CONJ'])
                    alt = self.config['ACOMP']

                rand = random.choice(range(1))
                rand3 = random.choice(range(4))
                options3 = {
                    0:f"at most",
                    1:f"at maximum",
                    2:"a maximum of",
                    3:"not more than"
                }
                options = {
                    0:f"How many {realiser.realise(subj)} must be {alt} in a row such that this event has a {get_prob()} of {options3[rand3]} {p}%?",
                }

                self.questions.append(('hard',options[rand]))
