import os
import csv
import json
import time

from sklearn.metrics import classification_report
import numpy as np


########################################################################
############## HELPER FUNCTIONS ########################################
########################################################################
def replace_wrong_values(spec_traits, trait, wrong_trait_value, correct_trait_value):
    """

    :param spec_traits:
    :param trait:
    :param wrong_trait_value:
    :param correct_trait_value:
    :return:
    """
    if trait in spec_traits:
        if wrong_trait_value in spec_traits[trait_key]:
            if correct_trait_value in spec_traits[trait_key]:
                if spec_traits[trait_key][wrong_trait_value] == '1' or spec_traits[trait_key][correct_trait_value] == '1':
                    spec_traits[trait_key][correct_trait_value] = '1'
            else:
                spec_traits[trait_key][correct_trait_value] = spec_traits[trait_key][wrong_trait_value]
            del spec_traits[trait_key][wrong_trait_value]

    return spec_traits


def replace_wrong_key(spec_traits, wrong_trait, correct_trait):
    """
    There are cases where the LLM returns the wrong trait name, e.g. color instead of colour.
    Use this function to fix these kind of inconsistencies. This function also checks if the correct trait
    is alreaady in the dictionary and does some merging of the values.

    :param spec_traits: the dictionary of traits for a species.
    :param wrong_trait: the name of the wrong trait in the dictionary.
    :param correct_trait: the correct trait that should be in the dictionary.

    :return:
    """
    if wrong_trait in spec_traits:
        if correct_trait not in spec_traits:
            spec_traits[correct_trait] = spec_traits[wrong_trait]
            del spec_traits[wrong_trait]
        else:
            for val in spec_traits[wrong_trait]:
                if val not in traits[correct_trait.title()]:
                    continue
                if val not in spec_traits[correct_trait]:
                    spec_traits[correct_trait][val] = spec_traits[wrong_trait][val]
                else:
                    if spec_traits[wrong_trait][val] == '1' or spec_traits[correct_trait][val] == '1':
                        spec_traits[correct_trait][val] = '1'

            del spec_traits[wrong_trait]

    return spec_traits


def del_wrong_value(spec_traits, trait, wrong_trait_value):
    """
    Delete potential wrong value in the obtained dictionary of traits.
    LLMs tend to introduce some values or misspell some. In the latter, maybe check if you should
    replace instead of delete.

    :param spec_traits: the dictionary of traits for the species
    :param trait: the current trait in which the given wrong value is present
    :param wrong_trait_value: the wrong trait value to delete from the dictionary

    :return:
    """
    if wrong_trait_value in spec_traits[trait]:
        del spec_traits[trait][wrong_trait_value]

    return spec_traits


#####################################################
################## MAIN LOGIC #######################
#####################################################
if __name__ == '__main__':

    # set the path to write the results
    results_path = '../Results/Mistral/West'

    # First read the ground truth traits
    traits_path = '../Data/Traits/West.json'
    with open(traits_path, 'r') as f:
        traits = json.load(f)

    # Make a header for the csv file as was the original
    header_traits = [' ']
    header_values = [' ']
    for key in traits:

        # remove measurement trait from Palms
        if key == 'Measurement':
            continue
        for value in traits[key]:
            header_values.append(value.lower())
            header_traits.append(key.lower())

    # Read the ground truth summary and build a dictionary to compare to the inferred one from the LLM
    gt_traits_path = '../Data/DataFrames/DF_West.csv'

    # this contains all the values for the traits as a list for each species
    dict_gt = {}

    # a dicitonary form for easier access to trait names and values for each species
    species_wide_gt_traits = {}

    # this is the file
    with open(gt_traits_path, 'r') as f:

        reader = csv.reader(f)
        count = 0

        for row in reader:
            # the first three lines are description stuff
            if count < 3:
                count += 1
                continue

            # now it's a simple splitting
            spec_name = row[0]

            # parse the whole row just to get the values in a list
            dict_gt[spec_name] = [int(float(val)>0.) for val in row[1:]]

            # init the dict entry and parse
            species_wide_gt_traits[spec_name] = {}
            val_count = 1

            # the lower, title stuff is to match case between the files
            for idx, tr in enumerate(header_traits[1:]):
                tr = tr.lower()
                if tr not in species_wide_gt_traits[spec_name]:
                    species_wide_gt_traits[spec_name][tr] = {}
                    for val in traits[tr]:
                        val = val.lower()
                        species_wide_gt_traits[spec_name][tr][val] = int(row[val_count])
                        val_count += 1
                else:
                    continue


    # this walk can help us check if we have data for the species from the ground truth
    dir_list = next(os.walk(results_path))[1]

    # we are going to build a file similar to the original for easy comparison
    rows = []
    species_wide_llm_traits = {}
    with open('{}/summary.csv'.format(results_path), 'w', encoding='UTF-8', newline='') as csvf:

        writer = csv.writer(csvf)

        # write the header stuff first
        writer.writerow(header_traits)
        writer.writerow(header_values)
        writer.writerow(['Species'])

        # each folder is a species
        i = 0

        # you can sort first if you want
        # species_keys = sorted(list(dict_gt.keys()))
        for species in dict_gt.keys():
            cur_spec = species
            species = species.replace(' ', '_')

            # if the species is not in the folder list, we didn't have data for that
            # unless something went wrong with the script
            if species not in dir_list:
                print('Spec: {} not in dir list.'.format(species))
                writer.writerow([cur_spec] + ['-']*(len(header_values)-1))
                continue

            # now read the contents_all file that was saved from the response of the LLM
            # when we were trying MISTRAL, the json functionality was not working, so we need to do some
            # parsing. This needs to be changed depending on what you ask the LLM output to be.
            print('{}/{}/contents_all.txt'.format(results_path, species))
            with open('{}/{}/contents_all.txt'.format(results_path, species), 'r') as f:
                data = f.read().replace('(', '[').replace(')', ']')
                content_as_json = json.loads(data)

            # the code after the next code line performs some checks assessing if the traits and the values are correct
            # there are two options: 1) delete the wrong traits/values, ignore them, or do replacements.
            # thus, you can use the functions in the first lines to do that and recheck.
            # content_as_json = replace_wrong_key(content_as_json, 'Sepals calyx number', 'Sepals calyx numer')

            # now we are parsing the response to get the traits and their values
            # while checking at the same time that everything is valid.
            # spec traits will contain all the correct traits/values for the current species.
            spec_traits = {}
            for key in content_as_json:

                # this checks if the trait name is in the ground truth
                # here we ignore the wrong values and don't add it to the spec_traits dict
                # you can raise the nameerror to see what's wrong and add some lines before the loop to correct it.
                if key.lower() not in species_wide_gt_traits[cur_spec]:
                    #print(species_wide_gt_traits[cur_spec].keys())
                    print('Trait: \"{}\" not in the ground truth.'.format(key))
                    continue
                    #raise NameError('Trait: \"{}\" not in the ground truth.'.format(key))

                # if trait is correct put it in the dict.
                if key.lower() not in spec_traits:
                    spec_traits[key.lower()] = {}

                # now we check how the values are formatted. You may get values as strings, tuples, lists.
                # we want to build a dict after that.
                if isinstance(content_as_json[key], list):

                    for trait_values in content_as_json[key]:

                        # first check the format
                        if isinstance(trait_values, str):
                            trait_value, presence = eval(trait_values)
                        elif isinstance(trait_values, list) or isinstance(trait_values, tuple):
                            trait_value, presence = trait_values
                        else:
                            raise ValueError('Wrong format: {}'.format(trait_values))

                        # here check if trait value is not in the gt values
                        # in our case, we ignore it, but you can raise the nameerror and fix it before the trait loop.
                        if trait_value.lower() not in species_wide_gt_traits[cur_spec][key.lower()]:
                            #print(species_wide_gt_traits[cur_spec][key.lower()])
                            print('Trait value: \"{}\" for key: \"{}\" is not in the gt'.format(trait_value, key))
                            continue
                            # raise NameError('Trait value: \"{}\" for key: \"{}\" is not in the gt'.format(trait_value, key))

                        # if the value is not in the dict, add it
                        # if for some reason it is, check the value.
                        if trait_value not in spec_traits[key.lower()]:
                            spec_traits[key.lower()][trait_value.lower()] = presence
                        elif spec_traits[key.lower()][trait_value.lower()] == 1:
                            continue
                        else:
                            spec_traits[key.lower()][trait_value.lower()] = presence
                else:
                    raise ValueError('Wrong format..')

            # This is the final check.
            # Take all the traits from the ground truth and make sure it's there.
            # do the same for the values.
            # if some value is missing, add it with a zero value.
            for key in species_wide_gt_traits[cur_spec]:

                if key not in spec_traits:
                    raise NameError('Trait: \"{}\" not found in the constructed species traits.'.format(key))

                for val in species_wide_gt_traits[cur_spec][key]:
                    if val not in spec_traits[key]:
                        print('Value: \"{}\" of trait: \"{}\" not found in '
                                        'the constructed species traits.'.format(val, key))
                        # didn't find it through the LLM, add it with a zero value
                        spec_traits[key][val] = 0

            # if you get no output from the above, we can build the final list of values
            # build the list of elements to write to csv. First is species names.
            cur_values = [cur_spec]
            for key in spec_traits:
                for val in spec_traits[key]:
                    cur_values.append(spec_traits[key][val])

            # write to csv
            writer.writerow(cur_values)
            rows.append(cur_values)

            # add the species dict to the general llm trait dictionary
            species_wide_llm_traits[species.replace('_', ' ')] = spec_traits

    # make an easier to access dict from the above stuff
    dict_llm = {}
    for row in rows:
        dict_llm[row[0]] = [int(val) for val in row[1:]]

    ##########################################
    ####### FIND THE N/As PER SPECIES ########
    ##########################################
    # N/As per species
    nas_per_species = {}
    perc_found_trait = {}

    for key in species_wide_gt_traits:

        # we didn't have data for that species or something went wrong
        if key not in species_wide_llm_traits:
            print("Species {} not found in the LLM dictionary..".format(key))
            time.sleep(0.5)
            continue

        # find the N/As
        nas_per_species[key] = []

        # here take the gt traits to filter out empty values
        for idx, tr in enumerate(species_wide_gt_traits[key]):

            # this is a new trait, count the trait presence in the species
            if tr not in perc_found_trait:
                perc_found_trait[tr] = 0

            # this shouldn't happen
            if len(species_wide_llm_traits[key]) == 0:
                nas_per_species[key].append(tr)
                raise ValueError('Species: \"{}\". This should not happen.'.format(key))

            # see if we found an active value.
            count = 0.
            for val in species_wide_llm_traits[key][tr]:
                if species_wide_llm_traits[key][tr][val] == 1:
                    count += 1

            # if count is zero, all values were 0, add it to the N/A list for the species.
            if count == 0.:
                nas_per_species[key].append(tr)

            # found a value for this trait. Add it to the count of the found traits
            if count > 0:
                perc_found_trait[tr] += 1 / len(species_wide_llm_traits)

    # write the results to file for each trait.
    header = ['Trait', 'Presence Percentage']

    with open('{}/per_trait_percentage.csv'.format(results_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for key in perc_found_trait:
            row = [key, perc_found_trait[key]]
            writer.writerow(row)

    # now iterate over all species, note the N/A percentage
    # also keep a list with species where all traits are N/A
    spec_na_perc = {}
    all_NA = []

    for spec in nas_per_species:
        na_num = len(nas_per_species[spec])

        # the -1 here is because we remove the measurement
        spec_na_perc[spec] = na_num / (len(traits.keys()))

        # no info was found for the species
        if spec_na_perc[spec] == 1.:
            all_NA.append(spec)


    ##################################################################
    ############ METRICS INFORMATION PER SPECIES PER TRAIT ###########
    ##################################################################
    traits_stats_per_species = {}
    for key in species_wide_gt_traits:

        # we didn't have data for that species or something went wrong
        if key not in species_wide_llm_traits:
            print("Species {} not found in the LLM dictionary..".format(key))
            time.sleep(0.5)
            continue

        # add the species to the dict
        if key not in traits_stats_per_species:
            traits_stats_per_species[key] = {}

        # parse all the traits for each species
        for tr in species_wide_gt_traits[key]:
            values_gt = list(species_wide_gt_traits[key][tr].values())
            values_llm = list(species_wide_llm_traits[key][tr].values())

            # ok if we didn't find a value for the trait, set the metrics to nan
            if tr in nas_per_species[key]:
                traits_stats_per_species[key][tr] = {}
                traits_stats_per_species[key][tr]['precision'] = np.nan
                traits_stats_per_species[key][tr]['recall'] = np.nan
                traits_stats_per_species[key][tr]['f1-score'] = np.nan

            else:
                metrics_trait = classification_report(values_gt, values_llm, output_dict=True)

                traits_stats_per_species[key][tr] = {}
                traits_stats_per_species[key][tr]['precision'] = metrics_trait['macro avg']['precision']
                traits_stats_per_species[key][tr]['recall'] = metrics_trait['macro avg']['recall']
                traits_stats_per_species[key][tr]['f1-score'] = metrics_trait['macro avg']['f1-score']

    # write the results to a file so build the headers first.
    header = ['Traits']
    subheader = ['Species']

    for tr in traits:
        for i in range(3):
            header.append(tr)
        subheader.extend(['Precision', 'Recall', 'F1'])

    with open('{}/per_trait_stats.csv'.format(results_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(subheader)

        for key in traits_stats_per_species:
            row = [key]
            for tr in traits_stats_per_species[key]:
                row.extend([traits_stats_per_species[key][tr]['precision'],
                            traits_stats_per_species[key][tr]['recall'],
                            traits_stats_per_species[key][tr]['f1-score']
                            ])

            writer.writerow(row)


    ##############################################################################
    ################### METRICS PER SPECIES ######################################
    ##############################################################################
    # first get some classification report per species
    # we need to both with and without the N/As and ignore the species with all traits N/A.
    metrics_per_species = {}
    metrics_per_species_na = {}

    macro_results_per_species = {}
    macro_results_per_species_na = {}

    for spec in species_wide_llm_traits:

        # ignore species with all NAs
        if spec in all_NA:
            metrics_per_species[spec] = {'macro avg': {'precision': 0., 'recall': 0., 'f1-score': 0.}}
            metrics_per_species_na[spec] = {'macro avg': {'precision': 0., 'recall': 0., 'f1-score': 0.}}

            macro_results_per_species[spec] = metrics_per_species[spec]['macro avg']
            macro_results_per_species_na[spec] = metrics_per_species_na[spec]['macro avg']
            continue

        # without the n/as
        llm_presence_without_na = []
        gt_presence_without_na = []

        # this is with N/As
        gt_presence_full = []
        llm_presence_full = []

        for trait in species_wide_llm_traits[spec]:

            for val in species_wide_llm_traits[spec][trait]:
                llm_presence_full.append(species_wide_llm_traits[spec][trait][val])
                gt_presence_full.append(species_wide_gt_traits[spec][trait][val])

                if trait in nas_per_species[spec]:
                    continue
                else:
                    llm_presence_without_na.append(species_wide_llm_traits[spec][trait][val])
                    gt_presence_without_na.append(species_wide_gt_traits[spec][trait][val])

        metrics_per_species[spec] = classification_report(gt_presence_full, llm_presence_full, output_dict=True)
        metrics_per_species_na[spec] = classification_report(gt_presence_without_na, llm_presence_without_na,
                                                             output_dict=True)


        macro_results_per_species[spec] = metrics_per_species[spec]['macro avg']
        macro_results_per_species_na[spec] = metrics_per_species_na[spec]['macro avg']

    # write some jsons with the results, if someone wants to see more information
    with open('{}/metrics_per_species.json'.format(results_path), 'w') as f:
        json.dump(metrics_per_species, f)

    with open('{}/metrics_per_species_na.json'.format(results_path), 'w') as f:
        json.dump(metrics_per_species_na, f)

    with open('{}/macro_metrics_per_species.json'.format(results_path), 'w') as f:
        json.dump(macro_results_per_species, f)

    with open('{}/macro_metrics_per_species_na.json'.format(results_path), 'w') as f:
        json.dump(macro_results_per_species_na, f)

    # This is the final summary for the metrics per species.
    header = ['Species', 'All Traits', ' ',' ', 'Without N/As', '', '', 'N/A Percentage']
    subheader = [' ', 'Precision', 'Recall', 'F1-Score', 'Precision', 'Recall', 'F1-Score', ' ']

    with open('{}/metrics_per_species_summary_macro.csv'.format(results_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(subheader)

        for key in metrics_per_species:

            row = [key]
            row.extend([macro_results_per_species[key]['precision'],
                        macro_results_per_species[key]['recall'],
                        macro_results_per_species[key]['f1-score']])
            row.extend([macro_results_per_species_na[key]['precision'],
                        macro_results_per_species_na[key]['recall'],
                        macro_results_per_species_na[key]['f1-score']])
            row.append(spec_na_perc[key])
            writer.writerow(row)
