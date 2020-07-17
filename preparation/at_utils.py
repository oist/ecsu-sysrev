from airtable import Airtable
import json
from commons import jdump
from key import AIRTABLE_API_KEY

BASE_ID = 'app9E5X8IBbf60rws'

def get_AT(table_name):
    return Airtable(
        BASE_ID, 
        table_name, 
        api_key=AIRTABLE_API_KEY
    )

'''
Current required fields:
Source: <str>
Year: <int>, 
Authors: <str>, 
Title: <str>, 
Abstract: <str>
'''

def export_to_at():
    import mendeley_search
    import wos_search     
    accepted_mendeley_papers = mendeley_search.refine_results()
    accepted_wos_papers = wos_search.refine_results()
    AT = get_AT('Auto')
    AT.batch_insert([mendeley_search.normalize_entry(p) for p in accepted_mendeley_papers])
    AT.batch_insert([wos_search.normalize_entry(p) for p in accepted_wos_papers])

def annotate_manual():
    from commons import validate_paper
    AT = get_AT('Manual')
    all_records = AT.get_all()
    for r in all_records:
        fields = r['fields']
        title = fields['Title']
        abstract = fields.get('Abstract','')
        mod_on_matches, mod_off_matches, mod_sense_matches = validate_paper(title, abstract)
        new_fields = {
            'MOD_ON': '\n'.join(mod_on_matches),
            'MOD_OFF': '\n'.join(mod_off_matches),
            'MOD_SENSE': '\n'.join(mod_sense_matches)
        }
        AT.update(r['id'], new_fields)

def populate_quoted_searches():
    from collections import defaultdict
    search_hits = defaultdict(lambda: defaultdict(int))
    # search query -> engine_type  -> hits
    engine_info = {
        'Mendeley': {
            'file': 'data/mendeley_stats.json' 
        },
        'Google Scholar': {
            'file': 'data/google_scholar_stats.json' 
        },		
    }	

    for engine,v in engine_info.items():
        with open(v['file']) as f_in:
            query_info = json.load(f_in)
            for query, info in query_info.items():
                for type in ['TOTAL', 'FILTERED']:
                    engine_type = '{} {}'.format(engine,type)
                    search_hits[query][engine_type] = info[type]
                
    AT = get_AT('Quoted Searches')
    for q in search_hits:
        engine_hits = search_hits[q]
        record = dict(engine_hits)
        record['QUERY'] = q
        AT.insert(record)

def populate_aggregated_sampled():
    AT = get_AT('Aggregated Sampled')
    with open('data/aggregated_A_norm_sampled.json') as f_in:
        records = json.load(f_in)	
    for q,records in records.items():
        for r in records:
            AT.insert(
                {
                    'Title': r['Title'],
                    'Year': r['Year'],
                    'Authors': r['Authors'],
                    'Abstract': r['Abstract'],
                    'DOI': r['DOI'],
                    'MOD_ON': '\n'.join(r['mod_on_matches']),
                    'MOD_OFF': '\n'.join(r['mod_off_matches']),
                    'MOD_SENSE': '\n'.join(r['mod_sense_matches']),
                    'Engines': '\n'.join(r['engines']),
                }
            )

def download_manual():
    AT = get_AT('Manual')
    result = [r['fields'] for r in AT.get_all()]
    # print(json.dumps(result, indent=3, ensure_ascii=False))
    return result

def compute_agreement():
    from collections import defaultdict
    AT = get_AT('Aggregated Sampled')
    result = [r['fields'] for r in AT.get_all()]
    stats = defaultdict(int)
    for r in result:
        r_accepted_count = len([x for x in r if x.startswith('Relevant_')])
        stats[r_accepted_count] += 1
    print(json.dumps(stats, indent=3, ensure_ascii=False))

def download_annotated_aggregated_sampled_A():
    AT = get_AT('Aggregated Sampled A')
    result = [ ]
    selected_fields = ['Title', 'ID', 'Abstract']
    for row in AT.get_all():
        row_fields = row['fields']
        row_selected_fields = {k:v for k,v in row_fields.items() if k in selected_fields}
        row_selected_fields['Accepted'] = row_fields.get('Relevant_Katja',False)
        result.append(row_selected_fields)
    jdump(result, 'data/aggregated_A_annotated.json')

if __name__ == '__main__':
    # export_to_at()
    # annotate_manual()
    # populate_quoted_searches()
    # populate_aggregated_sampled()
    # compute_agreement()
    # download_manual()
    download_annotated_aggregated_sampled_A()
    