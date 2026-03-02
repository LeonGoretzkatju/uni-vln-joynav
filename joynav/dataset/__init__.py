import re

import joynav.utils.registry
from joynav.utils.registry import register_component
from joynav.dataset.streamvln_dataset import StreamVLNDataset
from joynav.dataset.continuous_action_dataset import ContinuousActionDataset
from joynav.dataset.vln_action_dataset import VLNActionDataset
from joynav.dataset.vln_discrete_action_dataset import VLNDiscreteActionDataset
from joynav.dataset.vln_discrete_action_geo_dataset import VLNDiscreteActionGeoDataset

register_component('dataset', 'streamvln', StreamVLNDataset)
register_component('dataset', 'continuous_action', ContinuousActionDataset)
register_component('dataset', 'vln_action', VLNActionDataset)
register_component('dataset', 'vln_discrete_action', VLNDiscreteActionDataset)
register_component('dataset', 'vln_discrete_action_geo', VLNDiscreteActionGeoDataset)
