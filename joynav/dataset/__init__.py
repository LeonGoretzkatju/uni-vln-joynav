import re

import joynav.utils.registry
from joynav.utils.registry import register_component
from joynav.dataset.vln_action_dataset import VLNActionDataset

register_component('dataset', 'vln_action', VLNActionDataset)
