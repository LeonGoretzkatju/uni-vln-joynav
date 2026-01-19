import re

import joynav.utils.registry
from joynav.utils.registry import register_component
from joynav.dataset.streamvln_dataset import StreamVLNDataset
from joynav.dataset.continuous_action_dataset import ContinuousActionDataset

register_component('dataset', 'streamvln', StreamVLNDataset)
register_component('dataset', 'continuous_action', ContinuousActionDataset)
