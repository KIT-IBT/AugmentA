import os
from dataclasses import asdict, fields
from typing import Optional

from config_types import (
    AugmentAConfig, RAFiberConfig, RAThresholds, RABundleSizes,
    PectinateMuscleConfig, RABachmannBundleConfig,
    LAFiberConfig, LAThresholds, LABundleSizes,
    BridgeConfig,
)

try:
    from augmentaupgrade.logging_setup import logger
except ImportError:
    from logging_setup import logger

config_file_location = 'augmenta_config.yaml'
_instance: Optional[AugmentAConfig] = None


def get_config() -> AugmentAConfig:
    global _instance, config_file_location
    if _instance is None:
        if os.path.exists(config_file_location):
            load_config()
        else:
            _instance = AugmentAConfig()
    return _instance


def save_config(filepath=None):
    global _instance
    if _instance is None:
        _instance = AugmentAConfig()
    if filepath is None:
        filepath = config_file_location
    from ruamel.yaml import YAML
    config_dic = asdict(_instance)
    _yaml = YAML()
    with open(filepath, 'w') as f:
        _yaml.dump(config_dic, f)
    logger.debug(f'Saved config to {filepath}')


def load_config(filepath=None):
    global _instance, config_file_location
    if filepath is None:
        filepath = config_file_location
    from ruamel.yaml import YAML
    _yaml = YAML()
    with open(filepath, 'r') as f:
        config_dic = _yaml.load(f)
    logger.debug(f'Loaded config from {filepath}')

    def safe_init(cls, dic):
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dic.items() if k in valid_keys})

    ra_dic = config_dic.pop('ra', {})
    la_dic = config_dic.pop('la', {})
    bridges_dic = config_dic.pop('bridges', {})

    _instance = AugmentAConfig(
        ra=RAFiberConfig(
            thresholds=safe_init(RAThresholds, ra_dic.pop('thresholds', {})),
            bundles=safe_init(RABundleSizes, ra_dic.pop('bundles', {})),
            pm=safe_init(PectinateMuscleConfig, ra_dic.pop('pm', {})),
            bb=safe_init(RABachmannBundleConfig, ra_dic.pop('bb', {}))
        ),
        la=LAFiberConfig(
            thresholds=safe_init(LAThresholds, la_dic.pop('thresholds', {})),
            bundles=safe_init(LABundleSizes, la_dic.pop('bundles', {}))
        ),
        bridges=safe_init(BridgeConfig, bridges_dic)
    )


def change_config_location(new_location):
    global config_file_location
    config_file_location = new_location