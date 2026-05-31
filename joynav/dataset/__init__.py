import re

import joynav.utils.registry
from joynav.utils.registry import register_component
from joynav.dataset.streamvln_dataset import StreamVLNDataset
from joynav.dataset.continuous_action_dataset import ContinuousActionDataset
from joynav.dataset.vln_action_dataset import VLNActionDataset
from joynav.dataset.no_interleave_vln_action_dataset import VLNActionDataset as NoInterleaveVLNActionDataset
from joynav.dataset.vln_action_spatial_forcing_dataset import VLNActionSpatialForcingDataset
from joynav.dataset.vln_action_omega_spatial_forcing_dataset import VLNActionOmegaSpatialForcingDataset
from joynav.dataset.continuous_vlnn1_action_dataset import (
    ContinuousActionMixedDataset,
    ContinuousActionMixedOmegaSpatialForcingDataset,
    ContinuousVLNN1ActionDataset,
    ContinuousVLNN1ActionInterleavedDataset,
    ContinuousVLNN1ActionInterleavedOmegaSpatialForcingDataset,
    ContinuousVLNN1ActionOmegaSpatialForcingDataset,
)
from joynav.dataset.vln_discrete_action_dataset import VLNDiscreteActionDataset
from joynav.dataset.vln_discrete_action_geo_dataset import VLNDiscreteActionGeoDataset

register_component('dataset', 'streamvln', StreamVLNDataset)
register_component('dataset', 'continuous_action', ContinuousActionDataset)
register_component('dataset', 'vln_action', VLNActionDataset)
register_component('dataset', 'vln_action_interleave', VLNActionDataset)
register_component('dataset', 'vln_action_nointerleave', NoInterleaveVLNActionDataset)
register_component('dataset', 'vln_action_sf', VLNActionSpatialForcingDataset)
register_component('dataset', 'vln_action_sf_omega', VLNActionOmegaSpatialForcingDataset)
register_component('dataset', 'continuous_vlnn1_action', ContinuousVLNN1ActionDataset)
register_component('dataset', 'continuous_vlnn1_action_sf_omega', ContinuousVLNN1ActionOmegaSpatialForcingDataset)
register_component('dataset', 'continuous_vlnn1_action_noninterleave', ContinuousVLNN1ActionDataset)
register_component('dataset', 'continuous_vlnn1_action_noninterleave_sf_omega', ContinuousVLNN1ActionOmegaSpatialForcingDataset)
register_component('dataset', 'continuous_vlnn1_action_interleave', ContinuousVLNN1ActionInterleavedDataset)
register_component('dataset', 'continuous_vlnn1_action_interleave_sf_omega', ContinuousVLNN1ActionInterleavedOmegaSpatialForcingDataset)
register_component('dataset', 'continuous_action_mixed', ContinuousActionMixedDataset)
register_component('dataset', 'continuous_action_mixed_sf_omega', ContinuousActionMixedOmegaSpatialForcingDataset)
register_component('dataset', 'vln_discrete_action', VLNDiscreteActionDataset)
register_component('dataset', 'vln_discrete_action_geo', VLNDiscreteActionGeoDataset)
