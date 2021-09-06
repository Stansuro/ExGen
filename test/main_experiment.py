import os
import re
import sys
import pandas as pd
from nltk.sem.logic import ExpectedMoreTokensException
import spacy
nlp = spacy.load("en_core_web_trf")
from numpy.lib.type_check import real
from simplenlg.features.NumberAgreement import NumberAgreement
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
#print(parentdir)

from src.exercise_definitions.binomial_exercise_v02 import BinomialExercise
from src.exercise_definitions.normal_exercise_v01 import NormalExercise
import pickle
import random
import numpy as np
from copy import deepcopy
import ilm.tokenize_util
import torch
from transformers import GPT2LMHeadModel,AutoModelForSequenceClassification, AutoTokenizer, logging,BertForNextSentencePrediction,BertTokenizer,BertConfig
from sentence_transformers import SentenceTransformer, util
logging.set_verbosity_error()
from ilm.infer import infill_with_ilm
from nltk.tokenize import sent_tokenize
import simplenlg as nlg
from simplenlg import Tense as t
from simplenlg import Feature as f
lexicon = nlg.Lexicon.getDefaultLexicon()
factory = nlg.NLGFactory(lexicon)
realiser = nlg.Realiser(lexicon)

def c_to_s(c):
    return str(realiser.realise(factory.createSentence(c)))

class InputDialogTree:
    def __init__(self):
        config = dict()
        conj=None
        acomp=None
        subj=None
        verb=None
        unit=None
        valrange=None
        discreteness_indicator=None

        # ------------------ Input Dialog Tree ------------------------

        # Get entity of interest
        while not subj:
            txt = input("""
What is the random variable in your exercise? (single word)
    """).strip().lower()
            doc = nlp(txt)
            if len(doc) == 1:
                nsubj = doc[0].lemma_
                subj = factory.createNounPhrase(nsubj)
                if doc[0].lemma_!=doc[0].text:
                    subj.setPlural=True    # TODO fix, not working
            elif len(doc)==2:
                mod,comp=None,None
                for i,token in enumerate(doc):
                    if token.dep_=='ROOT':
                        nsubj = token.lemma_
                    if token.dep_=='compound':
                        if i==0:
                            mod='L'
                            comp=token.lemma_
                        else:
                            mod='R'
                            comp=token.lemma_
                subj = factory.createNounPhrase(nsubj)
                if mod=='L' and comp:
                    subj.setPreModifier(comp)
                if mod=='R' and comp:
                    subj.setPostModifier(comp)
            else:
                print('Currently only two random variables consisting of two words are supported!')

        n = deepcopy(subj)
        n.setDeterminer('the')

        # Decide whether the entities attributes are of interest or its actions
        while not discreteness_indicator:
            discreteness_indicator = input("""
Does your random variable take discrete or continuous values? (D/C)
    """).strip()
            if discreteness_indicator not in ['D','C']:
                discreteness_indicator=None

        if discreteness_indicator=='D':
            verb = 'is'

            # Get amount of distinct outcomes
            outcome_amount=None
            while not outcome_amount:
                outcome_amount = input("""
What is the amount of possible distinct values? (numeric value)
    """).strip()
                if not outcome_amount.isnumeric():
                    outcome_amount=None
                    continue
                if outcome_amount.isnumeric() and int(outcome_amount)!=2:
                    print('Currently only 2 distinct values are supported!')
                    outcome_amount=None

            # Get all distinct results
            # TODO: support more than 2 outcomes
            outcomes = []
            c = factory.createClause()
            c.setSubject(n)
            c.setVerb(verb)
            for i in range(int(outcome_amount)):
                outcome=None
                while not outcome:
                    txt=input(f"""
What is possible outcome no. {i+1}? (single word)
    """ + c_to_s(c)[:-1] + " ").strip().lower()
                    if txt=='' or txt is None:
                        outcome=None
                    else:
                        outcome=txt
                        outcomes.append(txt)

            if len(outcomes)==2:
                conj = outcomes[0]
                acomp = outcomes[1]
        else:

            verb = 'is'  
            while not unit:
                # Get Value unit
                txt=input("""
In which unit is """ + str(realiser.realise(n)) + """ being measured? (single word)
    """).strip().lower()
                if txt=='' or txt is None:
                    unit=None
                else:
                    unit = factory.createInflectedWord(lexicon.getWord(txt,nlg.LexicalCategory.NOUN),nlg.LexicalCategory.NOUN)
                    if len(txt) > 3 and not '/' in txt:
                        unit.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)

            while not valrange:
                # Get Value Range
                txt=input("""
What is the expected value of """ + str(realiser.realise(n)) + " in " + str(realiser.realise(unit)) + """? (integer)
    """).strip()
                if txt=='' or txt is None or not txt.isnumeric():
                    valrange=None
                else:
                    valrange=txt

        config['CONJ'] = conj
        config['ACOMP'] = acomp
        config['SUBJ'] = subj
        config['VERB'] = verb
        config['UNIT'] = unit
        config['VALRANGE'] = valrange

        self.config = config
        self.discreteness_indicator = discreteness_indicator

class GenerationController:
    def __init__(
        self,
        question_variability=1,       # TODO implement variability in exercise class (spinning)
        statement_variability=1,      # TODO implement variability in exercise class (spinning)
        max_statement_chunks=1,       # 1 -> All statements are generated in-line; n -> statements are split into at most n chunks and distributed in the exercise
        max_prefix_ext_sents=1,       # Maximum number of sentences to extend the prefix by
        max_infill_sents=1,           # Maximum number of consecutive infill constituents
        max_lookbehind=2,             # For successive infill only, consider up to n constituents to the left
        max_lookforward=2,            # For successive infill only, consider up to n constituents to the right
        successive_timeout=3,         # Max tries to generate addidional sentences in successive infill
        min_statement_chunks=1        # Minimum number of statement chunks
        ):
        self.question_variability = question_variability
        self.statement_variability = statement_variability
        self.max_statement_chunks = max_statement_chunks
        self.max_prefix_ext_sents = max_prefix_ext_sents
        self.max_infill_sents = max_infill_sents
        self.max_lookbehind = max_lookbehind
        self.max_lookforward = max_lookforward
        self.successive_timeout = successive_timeout
        self.min_statement_chunks = min_statement_chunks

    def get_arrangement(self, statements):
        assert len(statements) > 0, "No statements have been passed to arrangement generation!"
        assert len(statements) < 11, "Too many statements, up to 10 statements are currently supported!"
        
        arrangement = []
        arrangement += 'P'
        arrangement += 'I' * random.choice(range(1,self.max_prefix_ext_sents+1))
            
        rem_chunks = self.max_statement_chunks
        rem_statements = len(statements)
        while rem_chunks >= self.min_statement_chunks and rem_statements > 0 and rem_chunks>1:
            amt = random.choice(range(1,rem_statements+1))
            arrangement += 'S'*amt
            arrangement += 'I'*random.choice(range(1,self.max_infill_sents+1))
            rem_chunks -= 1
            rem_statements -= amt
        while rem_statements > 0:
            if rem_chunks <= rem_statements:
              arrangement += 'S'*rem_statements
              rem_chunks -= 1
              rem_statements = 0
            else:
              arrangement += 'S'
              arrangement += 'I'*random.choice(range(1,self.max_infill_sents+1))
              rem_statements -= 1
              rem_chunks -= 1
        while arrangement[-1] == 'I':
            arrangement = arrangement[:-1]
        
        arrangement = list(arrangement)
        i = 0
        while 'S' in arrangement:
            arrangement[arrangement.index('S')] = str(i)
            i += 1
        
        print(arrangement)
        return arrangement
    
    # TODO make successive generation consider what has previously generated by dynamically updating context_sents
    def successive_infill(self,context,infill_fn, blank_str, word_infill=False):
        blank_str = blank_str[1:] + ' '

        infills = []
        context_sents = sent_tokenize(context)
        for i in range(len(context_sents)):
            if blank_str in context_sents[i]:
                infill_num = context_sents[i].count(blank_str)
                start = max(i-(self.max_lookbehind-word_infill),0)
                end = i+1
                for j in range(1,self.max_lookforward+1):
                    check = min(len(context_sents)-1,i+j)
                    if blank_str in context_sents[check] or word_infill:
                        break
                    else:
                        end = check+1
                
                t = 0
                g_sents = []
                while len(g_sents)<=len(context_sents[start:end]) and t<self.successive_timeout:
                    c = ' '.join(context_sents[start:end])
                    g = infill_fn(c,infill_num,word_infill)
                    g_sents = sent_tokenize(g)
                    t += 1
                
                if word_infill:
                    if len(g_sents[-(end-i)])>len(context_sents[i]) and len(g_sents)==len(context_sents[start:end]):    # Probe whether sentence has been expanded
                        context_sents[i] = g_sents[-(end-i)]
                    else:
                        context_sents[i] = context_sents[i].replace(blank_str,'<INFILL FAILED> ')   # TODO remove in final product
                else:
                    if len(g_sents)>len(context_sents[start:end]):  # Probe for newly generated sentences
                        infills.append((i,g_sents[i-start:-(end-i)]))
                        context_sents[i] = g_sents[-(end-i)]
                    else:
                        context_sents[i] = context_sents[i].replace(blank_str,'<INFILL FAILED> ')   # TODO remove in final product

        incr = 0     
        while infills:
            i, t_list = infills.pop(0)
            t = ' '.join(t_list)
            context_sents.insert(i+incr,t)
            incr+=1

        return ' '.join(context_sents)

    def instantiate(
        self,
        exercise,
        infill_fn,
        check_fn,
        coh_fn,
        coh_prompt_fn,
        prefix,
        config,
        question_hardness,
        blank_str=' _',
        wordblank_str = ' §'
        ):
        
        exercise_instance = exercise(config)
        statements, question_candidates = exercise_instance.statements,exercise_instance.questions        # Statements should all be used, questions can be sampled, TODO implement question hardness
        if question_hardness=='easy':
            questions = random.sample([q for h,q in question_candidates if h=='easy'],3)
            self.min_statement_chunks = 1
            self.max_statement_chunks = 1
        if question_hardness=='medium':
            questions = random.sample([q for h,q in question_candidates if h=='easy'],1)
            questions = questions + random.sample([q for h,q in question_candidates if h=='medium'],2)
            self.min_statement_chunks = 2
        if question_hardness=='hard':
            questions = random.sample([q for h,q in question_candidates if h=='easy'],1)
            questions = questions + random.sample([q for h,q in question_candidates if h=='medium'],1)
            questions = questions + random.sample([q for h,q in question_candidates if h=='hard'],2)
            self.min_statement_chunks = 2
            self.max_infill_sents = 2

        arrangement = self.get_arrangement(statements)
        #print(arrangement)
        if type(exercise_instance)==BinomialExercise:
            statements = random.sample(statements,len(statements))
        if type(exercise_instance)==NormalExercise:
            statements = statements[:1] + random.sample(statements[1:],len(statements[1:]))

        context = ''
        for e in arrangement:
            if e == 'P':
                context += prefix
            if e == 'I':
                context += blank_str
            if e.isnumeric():
                if random.choice(range(2)) == 0:
                    context += ' ' + statements[int(e)]   
                else:
                    context += ' ' + statements[int(e)]   
                    #context += wordblank_str + ' , ' + statements[int(e)][0].lower() + statements[int(e)][1:]    # Forced connection with ',' (only to be used with infill_word)
                    #context += wordblank_str + ' ' + statements[int(e)][0].lower() + statements[int(e)][1:]    # Open connection (only to be used with infill_ngram)
                    # TODO Optimize (ngram produces new sentences when it shouldnt, single words might be bad)

        expose = self.successive_infill(context,infill_fn,blank_str,word_infill=False)
        expose = self.successive_infill(expose,infill_fn,wordblank_str,word_infill=True).replace(' ,',',')  # Dirty fix

        # TODO replace invalid constituents 
        # TODO End exercise via infill?

        # sent tokenize and check mnli
        sents = sent_tokenize(expose)
        sents_filter = [s for s in sents if s not in statements]
        prefix_sents_amt = len(sent_tokenize(prefix))
        validity = check_fn(statements, sents_filter[prefix_sents_amt:])  # Ignore user supplied prefix for validity check
        #coherence = coh_fn(sents_filter, arrangement)
        _,coherence, _, cos = coh_fn(sents[prefix_sents_amt-1:],arrangement)
        #print(sents)
        cos_prompt = coh_prompt_fn(sent_tokenize(prefix),[i for i,j in zip(sents[prefix_sents_amt-1:],arrangement) if j=='I'])

        if not validity or not all(coherence) or not all(cos) or not all(cos_prompt):
            return None
        
        if any([(i in expose) for i in ['>', '<', '|']]):
            return None
        
        return expose, arrangement

def main(p_hardness='easy',p_nucleus=0.95, p_cosdist=0.4, p_nsplogit=0.7, p_conflict=(0.5,0.2)):
    threshold = 10000
    count = 0
    df = pd.read_csv('App/test_dataset.csv', sep=';')

    # =============== HELPERS ================================================
    def get_model():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        model.eval()
        model.to(device)
        return model

    def get_tokenizer():
        tokenizer = ilm.tokenize_util.Tokenizer.GPT2
        with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
                additional_ids_to_tokens = pickle.load(f)
                additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
                try:
                        ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
                except ValueError:
                        print('Already updated')
                        print(additional_tokens_to_ids)
        return tokenizer, additional_tokens_to_ids

    def infill_text(context, infill_count, word_infill):
        context_ids = ilm.tokenize_util.encode(context, tokenizer)
        for i in range(infill_count):
            if word_infill:
                context_ids[context_ids.index(_wordblank_id)] = additional_tokens_to_ids['<|infill_word|>']
            else:
                context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']        

        generation = infill_with_ilm(
                            model,
                            additional_tokens_to_ids,
                            context_ids,
                            num_infills=1,
                            nucleus=p_nucleus
                        )[0]    # TODO test temp and nucleus parameters
        return ilm.tokenize_util.decode(generation, tokenizer)

    def check_consecutive_coherence(sents, arrangement):
        coh = []
        cos = []
        for i in range(len(sents)-1):
            s1 = sents[i]
            s2 = sents[i+1]
            encoding = coh_tokenizer(s1, s2, return_tensors='pt')

            outputs = coh_model(**encoding, labels=torch.LongTensor([1]))
            logits = outputs.logits.softmax(dim=1)
            #print(logits)
            if logits[0, 0] < p_nsplogit:
                coh.append(False)
            else:
                coh.append(True)

            cos_dist = util.pytorch_cos_sim(cos_model.encode(s1, convert_to_tensor=True).to(device), cos_model.encode(s2, convert_to_tensor=True).to(device))
            if cos_dist.item() < p_cosdist:
                cos.append(True)
            else:
                cos.append(False)
                #print(str(cos_dist.item()) + " - <" + s1 + "><" + s2 +">")

        coh_adj = deepcopy(coh)
        cos_adj = deepcopy(cos)
        p = re.compile(r'[IP]\d')
        for i in p.finditer(''.join(arrangement)):
            if i.span()[0]<len(coh_adj):      # Dirty fix
              coh_adj[i.span()[0]] = True   # Do not check statement to statement coherence
              cos_adj[i.span()[0]] = True
        p = re.compile(r'(?=(\d{2}))')
        for i in p.finditer(''.join(arrangement)):
            if i.span()[0]<len(coh_adj):      # Dirty fix
              coh_adj[i.span()[0]] = True   # Do not check statement to statement coherence
              cos_adj[i.span()[0]] = True
        return coh,coh_adj, cos, cos_adj
    
    def check_total_coherence(filtered_sents, arrangement):
        coh=[]
        for s1 in filtered_sents:
            for s2 in filtered_sents:
                encoding = coh_tokenizer(s1, s2, return_tensors='pt')
                outputs = coh_model(**encoding, labels=torch.LongTensor([1]))
                logits = outputs.logits
                if logits[0, 0] < logits[0, 1]:
                    coh.append(False)
                else:
                    coh.append(True)

        return all(coh)

    def check_coherece_to_prompt(prompt, candidates):
        cos = []
        #print(prompt)
        #print(candidates)
        for i,c in enumerate(candidates):
            for s in prompt:
                cos_dist = util.pytorch_cos_sim(cos_model.encode(s, convert_to_tensor=True).to(device), cos_model.encode(c, convert_to_tensor=True).to(device))
                if cos_dist.item() < p_cosdist:
                    cos.append(True)
                else:
                    #print(cos_dist,c,s)
                    cos.append(False)
                
        return cos

    def check_constraint_conflicts(statements, candidates):
        conflict = False
        for i,c in enumerate(candidates):
            if conflict:
                break
            for s in statements:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                x = nli_tokenizer.encode(s, c, return_tensors='pt').to(device)
                probs = nli_model(x)[0][0].softmax(dim=0)

                entailment = probs[2].item()
                neutral = probs[1].item()
                contradiction = probs[0].item()

                conflict = contradiction > p_conflict[0] and entailment < p_conflict[1]
                if conflict:
                    #print(f"<{c}> conflicts with <{s}> with a probability of {round(contradiction,2)*100}%")
                    break
                
        return not conflict

    # =============== PARAMS =================================================

    MODEL_DIR = 'ilm-master/models'
    _blank_str = ' _'
    _wordblank_str = ' §'

    model = get_model()
    tokenizer, additional_tokens_to_ids = get_tokenizer()
    _blank_id = ilm.tokenize_util.encode(_blank_str, tokenizer)[0]
    _wordblank_id = ilm.tokenize_util.encode(_wordblank_str, tokenizer)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
    nli_model.eval()
    nli_model.to(device)
    nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')

    coh_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    coh_model = BertForNextSentencePrediction.from_pretrained('nsp2')
    coh_model.eval()
    #coh_model.to(device)
    cos_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
    cos_model.eval()
    cos_model.to(device)

    g = GenerationController(max_statement_chunks=2, max_infill_sents=1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    i = 0
    while i < threshold and count<100:
        config = {}
        if df.iloc[i % len(df.index), 4] == 'D':
            exercise_type = BinomialExercise
            config['CONJ'] = df.iloc[i % len(df.index), 5]
            config['ACOMP'] = df.iloc[i % len(df.index), 6]
        else:
            exercise_type = NormalExercise
            txt = df.iloc[i % len(df.index), 9]
            unit = factory.createInflectedWord(lexicon.getWord(txt,nlg.LexicalCategory.NOUN),nlg.LexicalCategory.NOUN)
            if len(txt) > 3 and not '/' in txt:
                unit.setFeature(f.NUMBER,nlg.NumberAgreement.PLURAL)
            config['UNIT'] = unit
            config['VALRANGE'] = df.iloc[i % len(df.index), 10]
        
        doc = nlp(df.iloc[i % len(df.index), 7])
        if len(doc) == 1:
            nsubj = doc[0].lemma_
            subj = factory.createNounPhrase(nsubj)
            if doc[0].lemma_!=doc[0].text:
                subj.setPlural=True    # TODO fix, not working
        elif len(doc)==2:
            mod,comp=None,None
            for j,token in enumerate(doc):
                if token.dep_=='ROOT':
                    nsubj = token.lemma_
                if token.dep_=='compound':
                    if j==0:
                        mod='L'
                        comp=token.lemma_
                    else:
                        mod='R'
                        comp=token.lemma_
            subj = factory.createNounPhrase(nsubj)
            if mod=='L' and comp:
                subj.setPreModifier(comp)
            if mod=='R' and comp:
                subj.setPostModifier(comp)
        config['SUBJ'] = subj
        config['VERB'] = df.iloc[i % len(df.index), 8]

        instance = g.instantiate(exercise_type, infill_text,check_constraint_conflicts,check_consecutive_coherence,check_coherece_to_prompt, df.iloc[i % len(df.index), 3], config, p_hardness, _blank_str, _wordblank_str)
        if instance:
            count += 1
            # =============== OUTPUTS ================================================
            with open('logs.csv','a+') as myfile:
                myfile.write(
                        str(i) + "," + \
                        str(count) + "," + \
                        str(df.iloc[i % len(df.index), 0]) + "," + \
                        str(df.iloc[i % len(df.index), 1]) + "," + \
                        str(df.iloc[i % len(df.index), 2]) + "," + \
                        "'" + p_hardness + "'" + "," + \
                        str(p_nucleus) + "," + \
                        str(p_cosdist) + "," + \
                        str(p_nsplogit) + "," + \
                        str(p_conflict) + "," + \
                        "'" + instance[0].replace("'",'"') + "'" + "," +
                        "'" + ''.join(instance[1]) + "'" + "\n"
                        )
            print('SUCCESS')

        # TODO test truecase
        print(f"{i+1}/X instances done...")
        i += 1


if __name__=='__main__':
    with open('logs.csv','a+') as myfile:
        myfile.write(
                "Iteration," + \
                "Count," + \
                "StoryID," + \
                "Num_Sents," + \
                "Num_Chars," + \
                "Hardness," + \
                "Nucleus," + \
                "CosDistThresh," + \
                "NSPLogitThresh," + \
                "ConflictThresh," + \
                "Text," + \
                "Arrangement\n"
                )
    for p_hardness in ['easy','medium','hard']:
        for p_nucleus in [0.95,0.99]:
            for p_cosdist in [.4,.2]:
                for p_nsplogit in [.7,.9]:
                    for p_conflict in [(.5,.2),(.7,.15)]:
                        main(p_hardness,p_nucleus,p_cosdist,p_nsplogit,p_conflict)
