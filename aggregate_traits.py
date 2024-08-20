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
        if wrong_trait_value in spec_traits[trait]:
            if correct_trait_value in spec_traits[trait]:
                if spec_traits[trait][wrong_trait_value] == '1' or spec_traits[trait][correct_trait_value] == '1':
                    spec_traits[trait][correct_trait_value] = '1'
            else:
                spec_traits[trait][correct_trait_value] = spec_traits[trait][wrong_trait_value]
            del spec_traits[trait][wrong_trait_value]

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
    if trait not in spec_traits:
        return spec_traits

    if wrong_trait_value in spec_traits[trait]:
        del spec_traits[trait][wrong_trait_value]

    return spec_traits

#####################################################
################## MAIN LOGIC #######################
#####################################################
def post_processing(traits_gt, species_gt, responses_paths):

    # Make a header for the csv file as was the original
    header_traits = [' ']
    header_values = [' ']

    # for all the traits keys
    for key in traits_gt:
        # for all the the trait values
        for value in traits_gt[key]:
            header_values.append(value)
            header_traits.append(key)

    traits_gt = {k.lower(): [val.lower() for val in v] for k, v in traits_gt.items()}

    # this walk can help us check if we have data for the species from the ground truth
    dir_list = next(os.walk(responses_paths))[1]

    # we are going to build a file similar to the original for easy comparison
    rows = []
    species_wide_llm_traits = {}
    with open(f'summary.csv', 'w', encoding='UTF-8', newline='') as csvf:

        writer = csv.writer(csvf)

        # write the header stuff first
        writer.writerow(header_traits)
        writer.writerow(header_values)
        writer.writerow(['Species'])

        # each folder is a species
        i = 0

        # you can sort first if you want
        # species_keys = sorted(list(dict_gt.keys()))
        for species in species_gt:
            cur_spec = species
            species = species.replace(' ', '_')

            # if the species is not in the folder list, we didn't have data for that
            # unless something went wrong with the script
            if species not in dir_list:
                print('Spec: {} not in dir list.'.format(species))
                writer.writerow([cur_spec] + ['-'] * (len(header_values) - 1))
                continue

            # now read the contents_all file that was saved from the response of the LLM
            # when we were trying MISTRAL, the json functionality was not working, so we need to do some
            # parsing. This needs to be changed depending on what you ask the LLM output to be.
            with open(f'{responses_paths}/{species}/contents_all.txt', 'r') as f:
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
                lower_key = key.lower()

                # this checks if the trait name is in the ground truth
                # here we ignore the wrong values and don't add it to the spec_traits dict
                # you can raise the nameerror to see what's wrong and add some lines before the loop to correct it.
                if lower_key not in traits_gt:
                    #print(species_wide_gt_traits[cur_spec].keys())
                    print('Trait: \"{}\" not in the ground truth. Ignoring..'.format(lower_key))
                    continue

                # if trait is correct put it in the dict.
                if lower_key not in spec_traits:
                    spec_traits[lower_key] = {}

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
                        if trait_value.lower() not in traits_gt[lower_key]:
                            #print(species_wide_gt_traits[cur_spec][key.lower()])
                            print('Trait value: \"{}\" for key: \"{}\" is not in the gt'.format(trait_value, key))
                            continue

                        # if the value is not in the dict, add it
                        # if for some reason it is, check the value.
                        if trait_value not in spec_traits[lower_key]:
                            spec_traits[lower_key][trait_value.lower()] = presence
                        elif spec_traits[lower_key][trait_value.lower()] == 1:
                            continue
                        else:
                            spec_traits[lower_key][trait_value.lower()] = presence
                else:
                    raise ValueError('Wrong format..')

            # This is the final check.
            # Take all the traits from the ground truth and make sure it's there.
            # do the same for the values.
            # if some value is missing, add it with a zero value.
            for key in traits_gt:

                if key not in spec_traits:
                    spec_traits[key] = {}
                    print('Trait: \"{}\" not found in the constructed species traits. Adding entry..'.format(key))

                for val in traits_gt[key]:
                    if val not in spec_traits[key]:
                        print('Value: \"{}\" of trait: \"{}\" not found in '
                                        'the constructed species traits.'.format(val, key))
                        # didn't find it through the LLM, add it with a zero value
                        spec_traits[key][val.lower()] = 0

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


