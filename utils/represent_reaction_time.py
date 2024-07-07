import numpy as np
from utils.rt_bins_options import options
from consts import STRATEGIC_FEATURES_ORDER


class ReactionTimeRep:
    """
    A class to represent and manage reaction time features in a configuration-based approach (instead of constants).

    Attributes:
    ----------
    reaction_time_bins : list of tuples
        A list of tuples representing the bins for reaction time intervals.
    reaction_time_columns_names : list of str
        A list of column names for reaction time features based on the specified representation (one-hot or ordinal).
    strategic_features_order : list of str
        A list combining strategic features with reaction time feature columns.
    strategy_dim : int
        The total number of features in the strategy dimension, which includes strategic features and reaction time features.

    Methods:
    -------
    __init__(config):
        Initializes the ReactionTimeRep with the given configuration.
    """

    def __init__(self, config):
        bins_option_num = config['rt_bins_option_num']
        columns_rep = config['rt_bins_columns_rep']
        self.reaction_time_bins = options[bins_option_num]
        if columns_rep == 'one-hot':
            self.reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in self.reaction_time_bins]
        elif columns_rep == 'ordinal':
            self.reaction_time_columns_names = [f"last_rection_time_category"]
        else: # both 
            self.reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in self.reaction_time_bins] + [f"last_rection_time_category"]

        self.strategic_features_order = STRATEGIC_FEATURES_ORDER + self.reaction_time_columns_names
        self.strategy_dim = len(self.strategic_features_order)

