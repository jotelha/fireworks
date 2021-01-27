# coding: utf-8

from __future__ import unicode_literals

"""
This module allows you to modify a dict (a spec) using another dict (an instruction).
The main method of interest is apply_dictmod().

This code is based heavily on the Ansible class of custodian <https://pypi.python.org/pypi/custodian>,
but simplifies it considerably for the limited use cases required by FireWorks.
"""

import copy
import json
import logging
import re

from monty.design_patterns import singleton

__author__ = "Shyue Ping Ong"
__credits__ = "Anubhav Jain"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__date__ = "Jun 1, 2012"


def _log_nested_dict(log_func, dct):
    for l in json.dumps(dct, indent=2, default=str).splitlines():
        log_func(l)


def dict_select(base_dct, selector_dct):
    """Select subset of nested base_dct by nested hierarchy marked by selector_dct.

    Args:
        base_dct: dict or list or anything
        selector_dict: dict or list or bool

   Returns:
        dct: same as base_dct,
            if nested dict or list, then only nested fields marked by selector dict of parallel structure
    """
    logger = logging.getLogger(__name__)
    if isinstance(selector_dct, dict):
        dct = {}
        for k, v in selector_dct.items():
            if k not in base_dct:
                logger.warning("{} not in base_dct '{}'.".format(k, base_dct))
            elif v is not False:
                logger.debug("Descending into sub-tree '{}' of '{}'.".format(
                    base_dct[k], base_dct))
                # descend
                dct[k] = dict_select(base_dct[k], v)
            else:
                logger.debug("Deselected sub-tree '{}' of '{}'.".format(
                    base_dct[k], base_dct))

    elif isinstance(selector_dct, list):  # base_dct and selector_dct must have same length
        logger.debug("Branching into element wise sub-trees of '{}'.".format(base_dct))
        dct = [dict_select(base, selector) for base, selector in zip(base_dct, selector_dct) if selector is not False]

    else:  # arrived at leaf, selected
        logger.debug("Selected value '{}'".format(base_dct))
        dct = base_dct

    return dct


# from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
def dict_inject(base_dct, injection_dct, add_keys=True):
    """ Recursively inject inject_dict into base_dict. Recurses down into dicts nested
    to an arbitrary depth, updating keys.

    Will not alter base_dct or injection_dct, but return a deep copy without references to any of the former.

    The optional argument ``add_keys``, determines whether keys which are
    present in ``injection_dct`` but not ``base_dict`` should be included in the
    new dict.

    Args:
        base_dct (dict): inject injection_dct into base_dct
        injection_dct (dict):
        add_keys (bool): whether to add new keys

    Returns:
        dct: constructed merge dict
    """
    logger = logging.getLogger(__name__)

    logger.debug("Inject 'injection_dct'...")
    _log_nested_dict(logger.debug, injection_dct)
    logger.debug("... into 'base_dct'...")
    _log_nested_dict(logger.debug, base_dct)

    if isinstance(injection_dct, dict) and isinstance(base_dct, dict):
        logger.debug("Treating 'base_dct' and 'injection_dct' as parallel dicts...")

        dct = copy.deepcopy(base_dct)
        # injection_dct = injection_dct.copy()
        for k, v in injection_dct.items():
            if k in base_dct and isinstance(base_dct[k], dict) and isinstance(v, dict):
                logger.debug("Descending into key '{}' for further injection.".format(k))
                dct[k] = dict_inject(base_dct[k], v, add_keys=add_keys)
            else:  # inject
                if k in base_dct:
                    logger.debug("Replacing dict item '{}: {}' with injection '{}'.".format(k, dct[k], injection_dct[k]))
                else:
                    logger.debug("Inserting injection '{}' at key '{}'.".format(injection_dct[k], k))
                dct[k] = copy.deepcopy(v)

    elif isinstance(injection_dct, list) and isinstance(base_dct, list) and (len(injection_dct) == len(base_dct)):
        logger.debug("Treating 'base_dct' and 'injection_dct' as parallel lists...")

        # in this case base_dct and injecion_dct must have same length
        dct = []
        for base, injection in zip(base_dct, injection_dct):
            if isinstance(base, dict) and isinstance(injection, dict):
                logger.debug("Descending into list item '{}' and injection '{}' for further injection.".format(
                             base, injection))
                dct.append(dict_inject(base, injection, add_keys=add_keys))
            else:
                logger.debug("Replacing list item '{}' with injection '{}'.".format(base, injection))
                dct.append(copy.deepcopy(injection))

    else:  # arrived at leaf, inject
        logger.debug("Treating 'base_dct' and 'injection_dct' as values.")
        logger.debug("Replacing '{}' with injection '{}'.".format(base_dct, injection_dct))
        dct = copy.deepcopy(injection_dct)

    return dct


def get_nested_dict(input_dict, key):
    current = input_dict
    toks = key.split("->")
    n = len(toks)
    for i, tok in enumerate(toks):
        if tok not in current and i < n - 1:
            current[tok] = {}
        elif i == n - 1:
            return current, toks[-1]
        current = current[tok]


def get_nested_dict_value(input_dict, key):
    """Uses '.' or '->'-splittable string as key to access nested dict."""
    if key in input_dict:
        val = input_dict[key]
    else:
        key = key.replace("->", ".")  # make sure no -> left
        split_key = key.split('.', 1)
        if len(split_key) == 2:
            key_prefix, key_suffix = split_key[0], split_key[1]
        else:  # not enough values to unpack
            raise KeyError("'{:s}' not in {}".format(key, input_dict))

        val = get_nested_dict_value(input_dict[key_prefix], key_suffix)

    return val


def set_nested_dict_value(input_dict, key, val):
    """Uses '.' or '->'-splittable string as key and returns modified dict."""
    if not isinstance(input_dict, dict):
        # dangerous, just replace with dict
        input_dict = {}

    key = key.replace("->", ".")  # make sure no -> left
    split_key = key.split('.', 1)
    if len(split_key) == 2:
        key_prefix, key_suffix = split_key[0], split_key[1]
        if key_prefix not in input_dict:
            input_dict[key_prefix] = {}
        input_dict[key_prefix] = set_nested_dict_value(
            input_dict[key_prefix], key_suffix, val)
    else:  # not enough values to unpack
        input_dict[key] = val

    return input_dict


def arrow_to_dot(input_dict):
    """
    Converts arrows ('->') in dict keys to dots '.' recursively.
    Allows for storing MongoDB neseted document queries in MongoDB.

    Args:
      input_dict (dict)

    Returns:
      dict
    """
    if not isinstance(input_dict, dict):
        return input_dict
    else:
        return {k.replace("->", "."): arrow_to_dot(v) for k, v in input_dict.items()}


@singleton
class DictMods(object):
    """
    Class to implement the supported mongo-like modifications on a dict.
    Supported keywords include the following Mongo-based keywords, with the
    usual meanings (refer to Mongo documentation for information):
        _inc
        _set
        _unset
        _push
        _push_all
        _add_to_set (but _each is not supported)
        _pop
        _pull
        _pull_all
        _rename

    However, note that "_set" does not support modification of nested dicts
    using the mongo {"a.b":1} notation. This is because mongo does not allow
    keys with "." to be inserted. Instead, nested dict modification is
    supported using a special "->" keyword, e.g. {"a->b": 1}
    """

    def __init__(self):
        self.supported_actions = {}
        for i in dir(self):
            if (not re.match('__\w+__', i)) and callable(getattr(self, i)):
                self.supported_actions["_" + i] = getattr(self, i)

    @staticmethod
    def set(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            d[key] = v

    @staticmethod
    def unset(input_dict, settings):
        for k in settings.keys():
            (d, key) = get_nested_dict(input_dict, k)
            del d[key]

    @staticmethod
    def push(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d:
                d[key].append(v)
            else:
                d[key] = [v]

    @staticmethod
    def push_all(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d:
                d[key].extend(v)
            else:
                d[key] = v

    @staticmethod
    def inc(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d:
                d[key] += v
            else:
                d[key] = v

    @staticmethod
    def rename(input_dict, settings):
        for k, v in settings.items():
            if k in input_dict:
                input_dict[v] = input_dict[k]
                del input_dict[k]

    @staticmethod
    def add_to_set(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d and (not isinstance(d[key], (list, tuple))):
                raise ValueError("Keyword {} does not refer to an array."
                                 .format(k))
            if key in d and v not in d[key]:
                d[key].append(v)
            elif key not in d:
                d[key] = v

    @staticmethod
    def pull(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d and (not isinstance(d[key], (list, tuple))):
                raise ValueError("Keyword {} does not refer to an array."
                                 .format(k))
            if key in d:
                d[key] = [i for i in d[key] if i != v]

    @staticmethod
    def pull_all(input_dict, settings):
        for k, v in settings.items():
            if k in input_dict and (not isinstance(input_dict[k], (list, tuple))):
                raise ValueError("Keyword {} does not refer to an array."
                                 .format(k))
            for i in v:
                DictMods.pull(input_dict, {k: i})

    @staticmethod
    def pop(input_dict, settings):
        for k, v in settings.items():
            (d, key) = get_nested_dict(input_dict, k)
            if key in d and (not isinstance(d[key], (list, tuple))):
                raise ValueError("Keyword {} does not refer to an array."
                                 .format(k))
            if v == 1:
                d[key].pop()
            elif v == -1:
                d[key].pop(0)


def apply_mod(modification, obj):
    """
    Note that modify makes actual in-place modifications. It does not
    return a copy.

    Args:
        modification:
            Modification must be {action_keyword : settings}, where action_keyword is a
            supported DictMod
        obj:
            A dict to be modified
    """
    for action, settings in modification.items():
        if action in DictMods().supported_actions:
            DictMods().supported_actions[action].__call__(obj, settings)
        else:
            raise ValueError("{} is not a supported action!".format(action))
