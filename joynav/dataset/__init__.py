import re

import joynav.utils.registry
from joynav.utils.registry import register_component
from joynav.dataset.streamvln_dataset import StreamVLNDataset

register_component('dataset', 'streamvln', StreamVLNDataset)
