import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import rc
from json.decoder import JSONDecodeError

rc('font',**{'family':'serif','serif':['Palatino'], 'size': 20})
rc('text', usetex=True)

surveys_gt = pd.read_csv('../Data/answers_surveys.csv')
responses_gt = list(surveys_gt['Result'].values)
traits_gt = list(surveys_gt['Main Trait'].values)
subtraits_gt = list(surveys_gt['GT Sub Traits'].values)

results_path = '../SurveyResults/Mistral/'

responses_gpt = []
gpt_found_values = []
with open(results_path + 'responses_mistral_surveys_cleaned.txt','r') as f:
    for idx, line in enumerate(f):
        line = line.replace('(','[').replace(')', ']')

        responses_gpt.append([])
        gpt_found_values.append(0.)  # 'Missing')
        
        try:
            cur_line = json.loads(line)

        except JSONDecodeError:
            raise ValueError(idx)
            continue

        keys = [key.lower() for key in cur_line.keys()]

        # check that we got the correct main trait as a response
        if traits_gt[idx].lower() not in keys:
            print(keys, traits_gt[idx])
            raise ValueError('Huh? Different Trait from the Original?')

        # get the entries for the specific trait. These should be lists (value, evidence)
        # thus, if any value has evidence = 1, we save it for further processing
        entries = cur_line[traits_gt[idx].capitalize()]
       
        for qv in entries:
            print(qv)
            if isinstance(qv, list):
                if  qv[1] == 1 and qv[0] in subtraits_gt[idx]:
                    responses_gpt[idx].append(qv)
                    gpt_found_values[idx] = 1.#'Found'
            elif isinstance(qv, dict):
                key = list(qv.keys())[0]
                if list(qv.values())[0]==1 and key in subtraits_gt[idx]:
                    responses_gpt[idx].append(qv)
                    gpt_found_values[idx] = 1.  # 'Found'
            else:
                raise ValueError('Something is wrong {}'.format(qv))

# iterate over the surveys results and the responses to see the true N/As
resp_aggregated = []
for idx, resp in enumerate(responses_gt):
    if resp in ['Can infer correct Entity', 'Can infer correct Quality', 'None of the above']:
        resp_aggregated.append(0.)
    elif resp == 'Can infer correct Value':
        resp_aggregated.append(1.)
    else:
        raise ValueError('Huh? Another kind of response? C est pas possible')

print(classification_report(resp_aggregated, gpt_found_values, output_dict = False, digits = 4))
corr = 0.
for i in range(len(gpt_found_values)):
    if resp_aggregated[i] == gpt_found_values[i]:# or responses_gpt[i] in responses_gt[i]:
        corr += 1.
    #else:
    #    print('GT: {}, GPT: {}'.format(responses_gt[i], responses_gpt[i]))
print('Correct: {}'.format(corr/len(gpt_found_values)))

conf_matr = confusion_matrix(resp_aggregated, gpt_found_values)

print(conf_matr)
plot = True
labels = ['LLM Missing', 'LLM Found']
y_labels = ['GT Missing', 'GT Found']
if plot:
    plt.figure(figsize = (7,5))
    axr =sn.heatmap(conf_matr, annot=True, fmt='g', xticklabels=labels, yticklabels=y_labels, cmap = 'OrRd')
    axr.tick_params(left = False, labeltop=True, bottom = False, labelbottom = False, rotation=0)
    plt.yticks(rotation=90)

    #plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_path + 'confusion_matrix_v2.pdf')