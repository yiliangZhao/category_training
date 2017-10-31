import time
import pandas as pd
import requests

def tokenization(text):
    try:
        result = requests.post('http://122.11.129.73:31200/queryproc?country=TW', data=text)
    except Exception as e:
        print e
        result = None
    if result:
        return result.text
    else:
        return ''


def distribute(all_lists, batch_size=5000):
    list_results = list()
    for i in range(int(len(all_lists) / batch_size)):
        list_results.append(all_lists[i * batch_size:i * batch_size + batch_size])
    if len(all_lists) % batch_size != 0:
        list_results.append(all_lists[batch_size * int(len(all_lists) / batch_size):])
    return list_results


def batch_tokenization(list_list_entities):
    list_results = list()
    for list_entities in list_list_entities:
        tokenized = tokenization('\n'.join(list_entities).encode('utf=8'))
        list_results.append(tokenized)
    return '\n'.join(list_results)


def remove_new_line(string):
    if type(string) != unicode:
        unicode_str = unicode(str(string), 'utf-8')
    else:
        unicode_str = string
    return unicode_str.replace('\n', ' ').replace('\r', ' ')


def tokenization_batch(df_small):
    """
    Perform batch tokenization
    :param df_small:
    :return: the same dataframe with three additional columns containing the list of tokens
    """

    df_small['name_nonewline'] = df_small['name'].apply(remove_new_line)
    df_small['description_nonewline'] = df_small['description'].apply(remove_new_line)

    # Remove possible Nan
    df_small['name_nonewline'] = df_small['name_nonewline'].apply(lambda x: '' if pd.isnull(x) else x)
    df_small['description_nonewline'] = df_small['description_nonewline'].apply(lambda x: '' if pd.isnull(x) else x)

    # print df_small[['itemid', 'description', 'description_nonewline']].head()
    all_name = list(df_small['name_nonewline'])
    all_description = list(df_small['description_nonewline'])

    list_tokenized = list()
    batch_sizes = [5000, 2500, 1250, 600, 300, 100, 50, 10]
    # try these batch sizes.

    # tokenization for item names
    for batch_size in batch_sizes:
        list_list_names = distribute(all_name, batch_size)
        list_tokenized = batch_tokenization(list_list_names)
        if len(list_tokenized.split('\n')) == len(df_small):
            break
    if (len(list_tokenized.split('\n')) != len(df_small)) and (len(df_small) > 0):
        df_small.to_csv('/tmp/category_item_name_' + str(time.time()) + '.csv', encoding='utf-8', index=None)
    df_small['tokened_name'] = list_tokenized.split('\n')

    # tokenization for item descriptions
    for batch_size in batch_sizes[3:]:
        list_list_description = distribute(all_description, batch_size)
        list_tokenized = batch_tokenization(list_list_description)
        if len(list_tokenized.split('\n')) == len(df_small):
            break
    if (len(list_tokenized.split('\n')) != len(df_small)) and (len(df_small) > 0):
        df_small.to_csv('/tmp/category_item_description' + str(time.time()) + '.csv', encoding='utf-8', index=None)
    df_small['tokened_description'] = list_tokenized.split('\n')
    return df_small
