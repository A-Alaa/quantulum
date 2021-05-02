#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""quantulum classifier functions."""

# Standard library
import re
import os
import json
import pickle
import logging

# Dependencies
import wikipedia
from stemming.porter2 import stem
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# Quantulum
from . import classes as c
from .classes import Reference as r


###############################################################################
def download_wiki(filename):
    """Download WikiPedia pages of ambiguous units."""
    ambiguous = [i for i in r.UNITS.items() if len(i[1]) > 1]
    ambiguous += [i for i in r.DERIVED_ENT.items() if len(i[1]) > 1]
    pages = {(j.name, j.uri) for i in ambiguous for j in i[1]}

    objs = []
    for num, page in enumerate(pages):

        obj = {'url': page[1]}
        obj['_id'] = obj['url'].replace('https://en.wikipedia.org/wiki/', '')
        obj['clean'] = obj['_id'].replace('_', ' ')

        print('---> Downloading %s (%d of %d)' % \
              (obj['clean'], num + 1, len(pages)))

        obj['text'] = wikipedia.page(obj['clean'], auto_suggest=False).content
        obj['unit'] = page[0]
        objs.append(obj)

    path = os.path.join(r.TOPDIR, filename)
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'w') as file:
        json.dump(objs, file, indent=4, sort_keys=True)

    print('\n---> All done.\n')


###############################################################################
def clean_text(text):
    """Clean text for TFIDF."""
    new_text = re.sub(r'[^\w\s]', ' ', text, re.UNICODE)

    new_text = [stem(i) for i in new_text.lower().split() if not
                re.findall(r'[0-9]', i)]

    new_text = ' '.join(new_text)

    return new_text


###############################################################################
def train_classifier(name_prefix, download=True, parameters=None, ngram_range=(1, 1)):
    """Train the intent classifier."""
    if download:
        download_wiki(f'{name_prefix}_wiki.json')

    print('\n---> Training..\n')
    path = os.path.join(r.TOPDIR, f'{name_prefix}_train.json')
    training_set = json.load(open(path)) if os.path.exists(path) else []
    path = os.path.join(r.TOPDIR, f'{name_prefix}_wiki.json')
    wiki_set = json.load(open(path))

    target_names = list({i['unit'] for i in training_set + wiki_set})
    train_data, train_target = [], []

    for example in training_set + wiki_set:
        train_data.append(clean_text(example['text']))
        train_target.append(target_names.index(example['unit']))

    with open(os.path.join(r.TOPDIR, '_debug_train_data.json'), 'w') as file:
        json.dump(train_data, file, indent=4, sort_keys=True)
    with open(os.path.join(r.TOPDIR, '_debug_target_names.json'), 'w') as file:
        json.dump(target_names, file, indent=4, sort_keys=True)

    tfidf_model = TfidfVectorizer(sublinear_tf=True,
                                  ngram_range=ngram_range,
                                  stop_words='english')

    matrix = tfidf_model.fit_transform(train_data)

    if parameters is None:
        parameters = {'loss': 'log', 'penalty': 'l2', 'max_iter': 50,
                      'alpha': 0.00001, 'fit_intercept': True}

    clf = SGDClassifier(**parameters).fit(matrix, train_target)
    obj = {'tfidf_model': tfidf_model,
           'clf': clf,
           'target_names': target_names}
    path = os.path.join(r.TOPDIR, f'{name_prefix}_clf.pickle')
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


###############################################################################
def load_classifier(name_prefix):
    """Train the intent classifier."""
    path = os.path.join(r.TOPDIR, f'{name_prefix}_clf.pickle')

    with open(path, 'rb') as file:
        obj = pickle.load(file)

    return obj['tfidf_model'], obj['clf'], obj['target_names']

###############################################################################
def disambiguate_entity(key, text):
    """Resolve ambiguity between entities with same dimensionality."""
    new_ent = r.DERIVED_ENT[key][0]

    if len(r.DERIVED_ENT[key]) > 1:
        transformed = r.TFIDF_MODEL.transform([text])
        scores = r.CLF.predict_proba(transformed).tolist()[0]
        scores = sorted(zip(scores, r.TARGET_NAMES), key=lambda x: x[0],
                        reverse=True)
        names = [i.name for i in r.DERIVED_ENT[key]]
        scores = [i for i in scores if i[1] in names]
        try:
            new_ent = r.ENTITIES[scores[0][1]]
        except IndexError:
            logging.debug('\tAmbiguity not resolved for "%s"', str(key))

    return new_ent


###############################################################################
def disambiguate_unit(unit, text):
    """
    Resolve ambiguity.

    Distinguish between units that have same names, symbols or abbreviations.
    """
    new_unit = r.UNITS[unit]
    if not new_unit:
        new_unit = r.LOWER_UNITS[unit.lower()]
        if not new_unit:
            raise KeyError('Could not find unit "%s"' % unit)

    if len(new_unit) > 1:
        transformed = r.TFIDF_MODEL.transform([clean_text(text)])
        scores = r.CLF.predict_proba(transformed).tolist()[0]
        scores = sorted(zip(scores, r.TARGET_NAMES), key=lambda x: x[0],
                        reverse=True)
        names = [i.name for i in new_unit]
        scores = [i for i in scores if i[1] in names]
        try:
            final = r.UNITS[scores[0][1]][0]
            logging.debug('\tAmbiguity resolved for "%s" (%s)', unit, scores)
        except IndexError:
            logging.debug('\tAmbiguity not resolved for "%s"', unit)
            final = new_unit[0]
    else:
        final = new_unit[0]

    return final
