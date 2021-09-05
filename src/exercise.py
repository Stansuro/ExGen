import simplenlg as nlg
from simplenlg import LexicalCategory as lexcat
from simplenlg import Feature as f
#from transformers import pipeline

class Exercise:
    def __init__(self,prefix="", config={}, statements=[], questions=[]):
        self.config = config
        self.statements = statements
        self.questions = questions
        self.statements_populated = []
        self.questions_populated = []
        self.prefix = prefix

    def populate(self):

        lexicon = nlg.Lexicon.getDefaultLexicon()
        realiser = nlg.Realiser(lexicon)

        for s in self.statements:
            self.statements_populated.append(str(realiser.realise(s)))
        for q in self.questions:
            self.questions_populated.append(str(realiser.realise(q)))
            


