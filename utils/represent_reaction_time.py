import numpy as np
from utils.rt_bins_options import options
from consts import STRATEGIC_FEATURES_ORDER


class ReactionTimeRep:
    def __init__(self, config):
        bins_option_num = config['rt_bins_option_num']
        columns_rep = config['rt_bins_columns_rep']
        self.reaction_time_bins = options[bins_option_num]
        if columns_rep == 'one-hot':
            self.reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in self.reaction_time_bins]
        elif columns_rep == 'ordinal':
            self.reaction_time_columns_names = [f"rection_time_category"]
        else:
            self.reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in self.reaction_time_bins] + [f"rection_time_category"]

        self.strategic_features_order = STRATEGIC_FEATURES_ORDER + self.reaction_time_columns_names
        self.strategy_dim = len(self.strategic_features_order)

