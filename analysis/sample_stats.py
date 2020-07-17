import pandas as pd
import sklearn.metrics
import statsmodels.stats.inter_rater
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pprint import pprint


# sample_annotation = pd.read_csv('Aggregated Sampled-Grid view.csv')
# sample_annotation.fillna(value={"Relevant_Katja": 'unchecked',
#                                 "Relevant_Lorena": 'unchecked',
#                                 "Relevant_Manuel": 'unchecked'}, inplace=True)
#
#
# num_rows = sample_annotation.shape[0]
# percentage_relevant_Katja = sum(sample_annotation['Relevant_Katja'] == 'checked')/num_rows*100
# percentage_relevant_Lorena = sum(sample_annotation['Relevant_Lorena'] == 'checked')/num_rows*100
# percentage_relevant_Manuel = sum(sample_annotation['Relevant_Manuel'] == 'checked')/num_rows*100
#
# top_100 = sample_annotation[:100]
# percentage_relevant_Katja_100 = sum(top_100['Relevant_Katja'] == 'checked')
# percentage_relevant_Lorena_100 = sum(top_100['Relevant_Lorena'] == 'checked')
# percentage_relevant_Manuel_100 = sum(top_100['Relevant_Manuel'] == 'checked')
#
# # for agreement between 2 people: Cohen's kappa
# lorena_katja_agreement = sklearn.metrics.cohen_kappa_score(
#     sample_annotation['Relevant_Katja'],
#     sample_annotation['Relevant_Lorena'])
#
# # for agreement between 3 or more people: Fleiss kappa
# checked_counts = top_100[['Relevant_Katja', 'Relevant_Lorena', 'Relevant_Manuel']].apply(
#     pd.Series.value_counts, axis=1)[['checked', 'unchecked']].fillna(0)
# lorena_katja_manuel_agreement = statsmodels.stats.inter_rater.fleiss_kappa(checked_counts)


# load gold sample tagging
sample_annotation = pd.read_csv('test_gold.csv', usecols=["ID", "DECISION 1", "CRITERION 1"],
                                nrows=100, dtype="object")

experts = ["chi", "federico", "kari", "katrina", "sofija", "thu", "delaney", "maritina"]
expert_results = dict()

for expert in experts:
    expert_results[expert] = dict()
    if expert == "chi":
        annotation = pd.read_csv("test_{}.csv".format(expert), delimiter=";",
                                 usecols=["Decision", "Check", "Criterion"], nrows=100, dtype="object")
    else:
        annotation = pd.read_csv("test_{}.csv".format(expert), usecols=["Decision", "Check", "Criterion"],
                                 nrows=100, dtype="object")

    annotation['Decision'].fillna(annotation['Check'], inplace=True)
    # annotation = annotation.astype({'Criterion': 'object'})

    if expert == "kari":
        annotation['Decision'] = annotation['Decision'].map({"YES": "accepted", "CHECK": "accepted"}).fillna(annotation["Decision"])
    elif expert in ["katrina", "sofija", "delaney"]:
        annotation['Decision'] = annotation['Decision'].map({"YES": "accepted"}).fillna(annotation["Decision"])
    else:
        annotation['Decision'] = annotation['Decision'].map({"yes": "accepted"}).fillna(annotation["Decision"])

    expert_results[expert]["Unfilled"] = annotation.isnull().sum()['Decision']
    annotation.fillna('unfilled', inplace=True)
    stripped = annotation['Decision'].apply(lambda x: x.strip())
    upper_cased = stripped.apply(lambda x: x.upper())
    sample_annotation[expert + "_d"] = upper_cased
    sample_annotation[expert + "_cr"] = annotation["Criterion"]

gold = sample_annotation["DECISION 1"]
# baseline = gold.copy(deep=True)
# baseline = baseline.map({"ACCEPTED": "REJECTED"})
# baseline.fillna("REJECTED", inplace=True)
# baseline_acc = accuracy_score(gold, baseline)
# print(baseline_acc)

for expert in experts:
    expert_results[expert]['accuracy'] = accuracy_score(gold, sample_annotation[expert + "_d"])
    # this returns in order tn, fp, fn, tp
    expert_results[expert]['confusion'] = confusion_matrix(gold,
                                                           sample_annotation[expert + "_d"],
                                                           labels=["REJECTED", "ACCEPTED"]).ravel()
    if expert != "sofija":
        expert_results[expert]['f1'] = f1_score(gold, sample_annotation[expert + "_d"], pos_label="ACCEPTED")

pprint(expert_results)

# check criterion overlap
sample_rejected = sample_annotation[sample_annotation['DECISION 1'] == "REJECTED"]
gold_cr = sample_rejected['CRITERION 1']

for expert in experts:
    print(expert)
    if expert != "chi":
        first_cr = sample_rejected[expert + "_cr"].apply(lambda x: x.split(",")[0])
        expert_results[expert]['criterion_accuracy'] = accuracy_score(gold_cr, first_cr)


majority_sample = sample_annotation[["DECISION 1", "chi_d", "federico_d", "kari_d",
                                     "katrina_d", "sofija_d", "thu_d", "delaney_d"]]
majority_sample.to_csv("all_decisions.csv")