from collections import defaultdict
import json
import string
import re
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)

PUNCTUATION = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~±©°’”·‘“«»•″∼≥' # removing dash

LEXICON = defaultdict(int)

PUNCT_CAT = '#PUCT#'
STOP_CAT = '#STOP#'
DIGIT_CAT = '#DIGIT#'
FOREIGN_CAT = '#FOREIGN#'
LEX_CATS = [PUNCT_CAT, STOP_CAT, DIGIT_CAT, FOREIGN_CAT]

def tokenize(text): 
    text = text.lower()
    text = re.sub(r'[-–—‐−]',' ',text)     
    raw_tokens = [t.text for t in tokenizer(text)]
    tokens = []
    for w in raw_tokens:
        assert w not in LEX_CATS
        if re.match(r'\s+',w):
            continue
        if w in PUNCTUATION:
            tokens.append(PUNCT_CAT)
        w = w.translate(str.maketrans('', '', PUNCTUATION))  # remove punctuation in words  
        if w == '':
            tokens.append(PUNCT_CAT) # word was made only of punctuations
        elif w in STOP_WORDS:
            tokens.append(STOP_CAT)
        elif re.match(r'\d+',w):
            tokens.append(DIGIT_CAT)
        elif re.match(r'[^a-z\d]',w):
            tokens.append(FOREIGN_CAT)
        else:
            tokens.append(w)
            LEXICON[w] += 1
    return tokens

def get_ngrams(l, n):
    for i in range(len(l) - n + 1):
        yield ' '.join(l[i:i+n])

def dump_lexicon():
    with open('data/discr_lex/lexicon.txt', 'w') as f_out:
        sorted_lexicon = sorted(LEXICON.items(), key=lambda x: -x[1])
        for l,f in sorted_lexicon:
            f_out.write('{}\t{}\n'.format(f,l))
    for f in range(6):
        print('lexicon freq>={}: {}'.format(f, sum(1 for k,v in LEXICON.items() if v>=f)))

def analize_lexicon():
    papers = json.load(open('data/aggregated_A_annotated.json'))    
    value_template = {
        'accepted': 0,
        'rejected': 0
    }
    accepted_texts, rejected_texts = [], []        
    for p in papers:
        title_abstract = '{} {}'.format(p.get('Title', ''), p.get('Abstract', ''))
        bin = accepted_texts if p['Accepted'] else rejected_texts
        bin.append(title_abstract)
    print('Accepted papers: {}'.format(len(accepted_texts)))
    print('Rejected papers: {}'.format(len(rejected_texts)))
    accepted_tokens = [tokenize(t) for t in accepted_texts]   
    rejected_tokens = [tokenize(t) for t in rejected_texts]   
    
    # write lexicon with frequncy to file
    dump_lexicon()
    
    max_ngram = 3
    ngrams_stats = [{} for _ in range(max_ngram)]
    for i in range(max_ngram):
        n = i+1
        gram_stats = ngrams_stats[i]
        for bin, update_field in [(accepted_tokens, 'accepted'), (rejected_tokens, 'rejected')]:
            for tokens in bin:
                for gram_str in get_ngrams(tokens, n):
                    if not any(c in gram_str for c in LEX_CATS):                
                        if gram_str in gram_stats:
                            entry = gram_stats[gram_str]                            
                        else:                            
                            entry = value_template.copy()
                            gram_stats[gram_str] = entry                        
                        entry[update_field] += 1
        for v in gram_stats.values():
            v['diff'] = v['accepted'] - v['rejected']
        with open('data/discr_lex/{}_grams.json'.format(n), 'w') as f_out:
            gram_stats_sorted = sorted(gram_stats.items(), key=lambda x: -x[1]['diff'])
            json.dump(gram_stats_sorted, f_out, indent=3, ensure_ascii=False)    

if __name__ == '__main__':
    analize_lexicon()
    