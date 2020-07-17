import json
import re
# from doi import get_clean_doi
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)
# import nltk
import csv

MOD_ON = [
    'active', 'active-guided', 'actively commanded', 
    'actively-guided', 'dynamic', 'dynamically-guided', 
    'effortful', 'scanned', 'self-generated'
]
MOD_OFF = [
    'passive', 'passive-guided', 'passively induced', 
    'passively-guided', 'static'
]
MODS = MOD_ON + MOD_OFF
SENSES = [
    # Perception
    'perceive', 'perceived', 'perceiving', 'perception', 'perceptual', 'perceptually',
    # Motion
    'exploration', 'exploratory movement', 'exploratory movements', 'kinaesthesis', 
    'kinaesthetic', 'motion', 'movement', 'movements', 'self-motion',
    # Visual
    'vision', 'visual', 'visually',
    # Auditory
    'acoustic', 'acoustically', 'auditory', 'hearing', 'listening',
    # Somatosensory
    'haptic', 'haptically', 'tactile', 'touch',
    # Olfactory
    'olfaction', 'olfactory', 'smell', 'smelling',
    # Gustatory
    'gustatory', 'taste', 'tasting'
]

MODS_SENSES_COMBO_PAIRS = [(a,n) for a in MODS for n in SENSES]
MODS_SENSES_COMBO_STR = ['{} {}'.format(a,n) for a,n in MODS_SENSES_COMBO_PAIRS]

IGNORE_PUNCT = ['"', '”', ')', '(', '-', '‘', '’', "'", ',', '.'] 
# e.g., active-vision == active vision
# this list was constructed to make the hard-constraint consistent with mendeley query search
# ',', '.' should be discarded for better filtering

NO_PUNCT_TOKENS = lambda s: [t.text for t in tokenizer(s) if t.text not in IGNORE_PUNCT]

MOD_ON_TOKENS = [NO_PUNCT_TOKENS(m) for m in MOD_ON]
MOD_OFF_TOKENS = [NO_PUNCT_TOKENS(m) for m in MOD_OFF]
MODS_SENSES_COMBO_TOKENS = [NO_PUNCT_TOKENS(p) for p in MODS_SENSES_COMBO_STR]

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return True
    return False

def validate_paper(title, abstract='', keywords_list=[]):
    keywords = '\n'.join(keywords_list)
    content = '\n'.join([title, abstract, keywords])
    content = content.lower()
    content_tokens = NO_PUNCT_TOKENS(content)
    mod_on_matches = [MOD_ON[i] for i,x in enumerate(MOD_ON_TOKENS) if contains(x,content_tokens)]
    mod_off_matches = [MOD_OFF[i] for i,x in enumerate(MOD_OFF_TOKENS) if contains(x,content_tokens)]
    # mod_sense_matches = any(x in nltk.bigrams(content_tokens) for x in MODS_SENSES_COMBO_PAIRS)
    mod_sense_matches = [
        MODS_SENSES_COMBO_STR[i] for i,x in enumerate(MODS_SENSES_COMBO_TOKENS) 
        if contains(x, content_tokens)
    ]
    return mod_on_matches, mod_off_matches, mod_sense_matches

def check_overlapping_mendeley():
    import json
    import mendeley_search
    import wos_search     
    import scopus_search
    import proquest_search
    accepted_mendeley_papers = json.load(open('data/accepted_mendeley.json'))
    accepted_wos_papers = json.load(open('data/accepted_wos.json'))
    accepted_scopus_papers = json.load(open('data/accepted_scopus.json'))
    accepted_proquest_papers = json.load(open('data/accepted_proquest.json'))
    
    mendeley_dois = set(mendeley_search.get_doi(p) for p in accepted_mendeley_papers)
    wos_dois = set(wos_search.get_doi(p) for p in accepted_wos_papers)
    scopus_dois = set(scopus_search.get_doi(p) for p in accepted_scopus_papers)
    proquest_dois = set(proquest_search.get_doi(p) for p in accepted_proquest_papers)
    for s in [mendeley_dois, wos_dois, scopus_dois, proquest_dois]:
        if None in s: s.remove(None)
    print("DOIs in Mendeley: {}".format(len(mendeley_dois)))
    print("DOIs in WoS: {}".format(len(wos_dois)))
    print("DOIs in Scopus: {}".format(len(scopus_dois)))
    print("DOIs in Proquest: {}".format(len(proquest_dois)))
    common_dois_mendeley_wos = sorted(mendeley_dois.intersection(wos_dois))
    common_dois_mendeley_scopus = sorted(mendeley_dois.intersection(scopus_dois))
    common_dois_mendeley_proquest = sorted(mendeley_dois.intersection(proquest_dois))
    print("Common DOIs Menedely-WOS: {}".format(len(common_dois_mendeley_wos)))
    print("Common DOIs Menedely-Scopus: {}".format(len(common_dois_mendeley_scopus)))
    print("Common DOIs Menedely-Proquest: {}".format(len(common_dois_mendeley_proquest)))
    # with open('./data/mendeley_wos_common_dois.json', 'w') as f_out:
    #     json.dump(common_dois_mendeley_wos, f_out, indent=3, ensure_ascii=False)

def refine_results(engine, ID_KEY, TITLE_KEY, ABSTRACT_KEY, KEYWORDS_KEY, normalize_func):
    import json
    from tqdm import tqdm
    from tqdm import trange
    print("Refining results for {}".format(engine))
    accepted_papers, accepted_papers_norm, discarded_papers = [], [], []
    stats = {}
    total_papers, total_unique_papers, no_abstract_count = 0, 0, 0
    id_set = set()
    for i in trange(len(MODS_SENSES_COMBO_STR), desc='reading files'):
        q = MODS_SENSES_COMBO_STR[i]
        file_path = './data/{}/{}.json'.format(engine, q)
        with open(file_path) as f_in:
            results = json.load(f_in)			
            total_papers += len(results)
            num_accepted = 0
            for j in trange(len(results), desc=file_path):
                r = results[j]				
                id  = r[ID_KEY]
                if id in id_set:
                    continue
                id_set.add(id)
                total_unique_papers += 1
                title = r[TITLE_KEY]
                abstract = r.get(ABSTRACT_KEY, '')
                keywords_list = r.get(KEYWORDS_KEY, [])
                if ABSTRACT_KEY not in r:
                    no_abstract_count += 1
                mod_on_matches, mod_off_matches, mod_sense_matches = validate_paper(title, abstract, keywords_list)
                if mod_on_matches and mod_off_matches and mod_sense_matches:
                    num_accepted += 1
                    norm_r = normalize_func(r)
                    norm_r['mod_on_matches'] = mod_on_matches
                    norm_r['mod_off_matches'] = mod_off_matches
                    norm_r['mod_sense_matches'] = mod_sense_matches
                    accepted_papers.append(r)
                    accepted_papers_norm.append(norm_r)
                else:
                    discarded_papers.append(r)
            stats[q] = {
                'TOTAL': len(results),
                'FILTERED': num_accepted
            }
    with open('./data/accepted_{}.json'.format(engine), 'w') as f_out:
        json.dump(accepted_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_{}_norm.json'.format(engine), 'w') as f_out:
        json.dump(accepted_papers_norm, f_out, indent=3, ensure_ascii=False)
    with open('./data/discarded_{}.json'.format(engine), 'w') as f_out:
        json.dump(discarded_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/{}_stats.json'.format(engine), 'w') as f_out:
            json.dump(stats, f_out, indent=3, ensure_ascii=False)		
    discarded_papers = total_papers - len(accepted_papers)
    print("Total papers: {}".format(total_papers))    
    print("Total unique papers: {}".format(total_unique_papers))    
    print("Papers without abstract: {}".format(no_abstract_count))
    print("Discarded papers: {}".format(discarded_papers))
    print("Accepted papers: {}".format(len(accepted_papers)))
    return accepted_papers  

def validate_test():
    assert(validate_paper(
            title = 'Touch Can Be as Accurate as Passively-Guided Kinaesthesis in Length Perception'
        ))

def aggregate_results_A():
    import json
    from collections import defaultdict
    aggregated = defaultdict(list)
    # mod_senses -> results
    # engines = ['wos','mendeley','google_scholar']
    # sources = ['Web of Science','Mendeley','Google Scholar']
    engines = ['mendeley']
    sources = ['Mendeley']


    total_papers = 0
    for engine in engines:
        file = 'data/accepted_{}_norm.json'.format(engine)
        with open(file) as f_in:
            papers = json.load(f_in)
            total_papers += len(papers)
            for p in papers:
                # mode_senses = '\n'.join(p['mod_sense_matches'])
                # aggregated[mode_senses].append(p)
                upper_title = p['Title'].upper()
                aggregated[upper_title].append(p)
    
    count_n_engine = lambda c: len(
        [1 for v in aggregated.values() 
        if len(set([x['Source'] for x in v]))==c]
    )

    # potential problems:
    # - the same paper can appear twice in the same engine (e.g., google scholar)
    # - two different papers could have the same title

    with open('data/aggregated_A/aggregated_A.json', 'w') as f_out:
        json.dump(aggregated, f_out, indent=3, ensure_ascii=False)

    print("Total papers: {}".format(total_papers))
    print("Total unique titles: {}".format(len(aggregated)))
    print("Total titles in 1 engine: {}".format(count_n_engine(1)))
    print("Total titles in 2 engines: {}".format(count_n_engine(2)))
    print("Total titles in 3 engines: {}".format(count_n_engine(3)))

    aggregated_norm = defaultdict(list)
    # for each title select only one pub (giving priority to engines in order) 
    # create a dict mode_sense -> p
    for v in aggregated.values():
        selected = dict(next(p for s in sources for p in v if p['Source']==s ))
        selected['engines'] = sorted(set([x['Source'] for x in v]))
        mode_senses = '\n'.join(selected['mod_sense_matches'])
        aggregated_norm[mode_senses].append(selected)

    with open('data/aggregated_A/aggregated_A_norm.json', 'w') as f_out:
        json.dump(aggregated_norm, f_out, indent=3, ensure_ascii=False)
    
    import math
    import random
    sample_rate = 2/100
    aggregated_norm_sampled = {}
    for k,v in aggregated_norm.items():
        sample_num = math.ceil(sample_rate * len(v))
        aggregated_norm_sampled[k] = random.sample(v,sample_num)

    with open('data/aggregated_A/aggregated_A_norm_sampled.json', 'w') as f_out:
        json.dump(aggregated_norm_sampled, f_out, indent=3, ensure_ascii=False)

def normalize_doi(d, rejected_set):
    if d == None or d == '':
        return None
    d = d.replace('//','/')
    d = re.sub(r'http.+?\.org/','',d)
    if d.startswith('10'):
        return d.lower()
    rejected_set.add(d)
    return None

# clean string
def cs(s):
    import string
    if s==None or not s.strip():
        return None
    s = s.translate(str.maketrans('', '', string.punctuation))    
    s = re.sub(r'\s+',' ',s)
    return s.strip().lower()

def jdump(o, file_path):
    json.dump(
        o, open(file_path, 'w'),
        indent=True, ensure_ascii=False
    )

def make_list_from_authors(source, authors):
    if source in ['Mendeley', 'Scopus']:
        return [x.strip() for x in authors.split(',')]
    elif source in ['Proquest', 'Web of Science']:
        return [x.replace(', ', ' ').strip() for x in authors.split(';')]


def aggregate_results_B():
    import json
    import rispy
    from collections import defaultdict

    # mod_senses -> results
    engines = ['wos','scopus', 'proquest', 'mendeley']
    sources = ['Web of Science','Scopus','Proquest', 'Mendeley']

    total = 0
    filtered = 0    

    for engine in engines:        
        for type in ['accepted','discarded']:
            file = 'data/{}_{}.json'.format(type, engine)
            papers = json.load(open(file))
            total += len(papers)
            if type == 'accepted':
                filtered += len(papers)
                                    
    print('Total papers: {}'.format(total))
    print('Total filtered papers: {}'.format(filtered))

    filtered_with_title = 0
    filtered_with_abstract = 0
    filtered_with_title_and_abstract = 0

    for engine in engines:
        file = 'data/accepted_{}_norm.json'.format(engine)
        papers = json.load(open(file))
        filtered_with_title += sum(1 for p in papers if cs(p['Title']))
        filtered_with_abstract += sum(1 for p in papers if cs(p['Abstract']))
        filtered_with_title_and_abstract += sum(1 for p in papers if cs(p['Title']) and cs(p['Abstract']))
    
    print('Total filtered papers with title: {}'.format(filtered_with_title))
    print('Total filtered papers with abstract: {}'.format(filtered_with_abstract))
    print('Total filtered papers with title and abstract: {}'.format(filtered_with_title_and_abstract))


    ### TITLES, DOIS AND ABSTRACTS

    title_doi_papers = defaultdict(lambda: defaultdict(list)) # title -> doi -> papers
    title_abstract_papers = defaultdict(lambda: defaultdict(list)) # title -> abstract
    dois_set, dois_rejected_set = set(), set()
    title_set = set()
    papers_with_doi = 0
    aggregated_list = []    
    
    for engine in engines:
        file = 'data/accepted_{}_norm.json'.format(engine)
        papers = json.load(open(file))
        for p in papers:
            title = cs(p['Title'])
            abstract = cs(p['Abstract'])
            if title and abstract:
                doi = normalize_doi(p['DOI'], dois_rejected_set)                                                
                duplicated_doi = False
                if doi:
                    papers_with_doi += 1                    
                    if doi in dois_set:
                        duplicated_doi = True                    
                    dois_set.add(doi)                    
                    title_doi_papers[title][doi].append(p)                          
                if not title in title_set and not duplicated_doi:                 
                    p['DOI'] = doi
                    aggregated_list.append(p)
                title_abstract_papers[title][abstract].append(p)             
                title_set.add(title)
                
    jdump(sorted(dois_set), 'data/aggregated_B/aggregated_B_dois.json')
    jdump(sorted(dois_rejected_set), 'data/aggregated_B/aggregated_B_dois_rejected.json')

    print('Papers with DOIs: {}'.format(papers_with_doi))
    print('Unique DOIs: {}'.format(len(dois_set)))        
    print('Unique titles: {}'.format(len(title_set)))


    papers_with_same_title_and_different_dois = {
        title: sorted(doi_paper_list.keys()) # doi_paper_list
        for title, doi_paper_list in title_doi_papers.items()         
        if len(doi_paper_list)>1
    }

    print('Papers with same title and different dois: {}'.format(
        sum(len(v) for v in papers_with_same_title_and_different_dois.values()))
    )

    jdump(
        papers_with_same_title_and_different_dois, 
        'data/aggregated_B/aggregated_B_papers_with_same_title_and_different_dois.json'
    )

    
    papers_with_same_title_and_different_abstract = {
        title: sorted(abstract_papers.keys())
        for title, abstract_papers in title_abstract_papers.items()         
        if len(abstract_papers)>1
    }

    jdump(
        papers_with_same_title_and_different_abstract, 
        'data/aggregated_B/aggregated_B_papers_with_same_title_and_different_abstract.json',
    )

    ### EXPORT AGGREGATION TO JSON
    aggregated_list = [
        {k:(v.strip() if isinstance(v, str) else v) for k,v in x.items()} 
        for x in aggregated_list 
    ]
    aggregated_list = sorted(aggregated_list, key=lambda x: x['Title'])
    print('Final aggreagated list size: {}'.format(len(aggregated_list)))
    for s in sources:
        num_aggregated_from_s = sum(1 for x in aggregated_list if x['Source']==s)
        print('\t- From {}: {}'.format(s, num_aggregated_from_s))
    jdump(
        aggregated_list,
        'data/aggregated_B/aggregated_B.json',
    )

    ### EXPORT AGGREGATION TO RIS
    from uuid import uuid4
    papers_per_ris_file = 3000
    papers_ris_chunks = [
        aggregated_list[x:x+papers_per_ris_file] 
        for x in range(0, len(aggregated_list), papers_per_ris_file)
    ]
    for n,chunk in enumerate(papers_ris_chunks,1):
        ris_entries = []
        for p in chunk:
            ris_entries.append({
                'authors': make_list_from_authors(p['Source'], p['Authors']),
                'title': p['Title'],
                'year': p['Year'],
                'abstract': p['Abstract'],
                'doi': p['DOI'],
                'keywords': p['mod_on_matches'] + p['mod_off_matches'] + p['mod_sense_matches'],
                'id': p['Source'] + '_' + str(uuid4()) #str(p['Source_id'])
            })
        with open('data/aggregated_B/RIS/aggregated_B_{}.ris'.format(n), 'w') as ris_file:
            rispy.dump(ris_entries, ris_file)
        
    ### EXPORT TO CSV
    headers = ['Source', 'Year', 'Authors', 'Title', 'Abstract', 'DOI'] #'mod_sense_matches'
    with open('data/aggregated_B/aggregated_B.csv', 'w') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        for p in aggregated_list:
            writer.writerow([p[k] for k in headers])

def check_manual_coverage():
    import json
    from at_utils import download_manual
    manual_records = download_manual()	
    sources_list = ['wos','mendeley','google_scholar']	
    engines_list = ['Web of Science','Mendeley','Google Scholar']	
    discarded_title_keys_list = ['TI','title','title']
    for i,e in enumerate(sources_list):		
        engine = engines_list[i]
        print(engine)
        discarded_title_key = discarded_title_keys_list[i]
        relevant_titles_upper = set([r['Title'].upper() for r in manual_records if r.get(engine,False)])
        accepted_json_file = './data/accepted_{}_norm.json'.format(e)
        discarded_json_file = './data/discarded_{}.json'.format(e)
        accepted_records = json.load(open(accepted_json_file))
        discarded_records = json.load(open(discarded_json_file))
        acceptted_titles_upper = set([e['Title'].upper() for e in accepted_records])		
        discarded_titles_upper = set([e[discarded_title_key].upper() for e in discarded_records])		
        all_titles_upper = discarded_titles_upper.union(acceptted_titles_upper)
        accepted_intersection = acceptted_titles_upper.intersection(relevant_titles_upper)
        all_intersection = all_titles_upper.intersection(relevant_titles_upper)
        all_percentage = round(len(all_intersection)/len(relevant_titles_upper)*100,2)
        accepted_percentage = round(len(accepted_intersection)/len(relevant_titles_upper)*100,2)
        print('All ({}): {}/{} = {}%'.format(len(all_titles_upper), len(all_intersection),len(relevant_titles_upper), all_percentage))
        print('Accepted ({}): {}/{} = {}%'.format(len(acceptted_titles_upper),len(accepted_intersection),len(relevant_titles_upper), accepted_percentage))
        print()


if __name__ == '__main__':
    # validate_test()
    # check_overlapping_mendeley()
    # aggregate_results_A()
    # check_manual_coverage()
    aggregate_results_B()
    

    