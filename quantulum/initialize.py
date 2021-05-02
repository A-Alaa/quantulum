# -*- coding: utf-8 -*-

"""quantulum global variables initializer"""

# Standard library
import os
import logging

# Quantulum
from . import load as l
from . import classifier as clf
from . import classes as c
from .classes import Reference as r
from . import regex as qre

def init(category='generic', other_units_file=None, other_entities_file=None):

    def init_units_entities(category='generic'):
        if category == 'generic':
            entities_file = 'entities.json'
            units_file = 'units.json'
        elif category == 'ehr':
            entities_file = 'entities_ehr.json'
            units_file = 'units_ehr.json'
        elif other_units_file is not None and other_entities_file is not None:
            entities_file = other_entities_file
            units_file = other_units_file
        else:
            raise Exception(f'Unrecognized category: {category}')

        r.CATEGORY = category
        r.ENTITIES, r.DERIVED_ENT = l.load_entities(entities_file)
        (r.NAMES, r.UNITS, r.LOWER_UNITS,
         r.SYMBOLS, r.DERIVED_UNI) = l.load_units(units_file)

    def init_classifier():
        try:
            r.TFIDF_MODEL, r.CLF, r.TARGET_NAMES= clf.load_classifier(r.CATEGORY)
        except Exception as e:
            logging.debug(f'\t{e.__doc__}\n'
                        '\t{e.message}\n'
                        '\tError loading trained model. Retraining..')
            clf.train_classifier(r.CATEGORY)
            r.TFIDF_MODEL, r.CLF, r.TARGET_NAMES = clf.load_classifier(r.CATEGORY)

    init_units_entities(category)
    init_classifier()
    qre.REG_DIM = qre.get_units_regex()
    r.INITIALIZED = True


