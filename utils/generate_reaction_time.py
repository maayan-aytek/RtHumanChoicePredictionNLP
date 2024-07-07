import os
import pickle
import warnings
import numpy as np
from reaction_time_model import lost_cause
from consts import PROBA2GO_DICT, REVIEWS_DICT
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class ReactionTimeGenerator:
    """
    Generates reaction times using different methods: baseline, random, model-based, or heuristic.
    Attributes:
    ----------
    kwargs : dict
        Additional parameters for generator functions.
    trained_model : model or None
        Loaded model for 'model' method.
    model_features : list of str or None
        Feature names required by the trained model.
    method : str
        Method used for reaction time generation.
    bot_thresholds : dict
        Thresholds for bot categories.
    strategies_rt_baselines : dict
        Baseline reaction times for strategies.
    generator_func : function
        Function for generating reaction times based on the method.
    """
    def __init__(self, method, rt_model_file_name = None, **kwargs):
        """Initializes the generator with the specified method and model file."""
        self.kwargs = kwargs
        self.trained_model = None
        self.model_features = None
        self.method = method
        self.bot_thresholds = {3: 10, 0: 7, 2: 9, 5: 9, 59: 8, 19: 9, 132: 10, 23: 9, 107: 9, 43: 9, 17: 9, 93: 9}
        if rt_model_file_name:
            with open(os.path.join('sweep_models', rt_model_file_name), 'rb') as file:
                self.trained_model = pickle.load(file)
                self.model_features = self.trained_model.feature_names_in_
        self.strategies_rt_baselines = {"random": 500, "correct - oracle": 3770, "LLM stochastic (Language-based)": 2920, "history_and_review_quality (Trustful)": 2830}
        self._set_generator_function()

    def _set_generator_function(self):
        """Sets the appropriate function for generating reaction times."""
        if self.method == 'baseline':
            self.generator_func = self._baseline
        elif self.method == 'random':
            self.generator_func = self._random
        elif self.method == 'model':
            self.generator_func = self._model
        elif self.method == 'heuristic':
            self.generator_func = self._heuristic
        else:
            raise ValueError(f'Unexpected method: {self.method}')

    def extract_features(self, row):
        """Extracts features from the data row."""
        user_strategy_name = row['user_strategy_name']
        row['played_oracle'] = 0
        row['played_random'] = 0
        row['played_trustful'] = 0
        row['played_llm'] = 0
        if user_strategy_name == 'history_and_review_quality (Trustful)':
            row['played_llm'] = 1
        elif user_strategy_name == 'random':
            row['played_random'] = 1
        elif user_strategy_name == 'correct - oracle':
            row['played_oracle'] = 1
        elif user_strategy_name == 'LLM stochastic (Language-based)':
            row['played_llm'] = 1
        else:
            raise ValueError(f'Unexpected strategy: {user_strategy_name}')
        
        review_id = row['reviewId']
        row['review_length'] = REVIEWS_DICT[review_id]['review_length']
        row['word_count'] = REVIEWS_DICT[review_id]['word_count']
        row['positive_review_proportion'] = REVIEWS_DICT[review_id]['positive_review_proportion']
        row['negative_review_proportion'] = REVIEWS_DICT[review_id]['negative_review_proportion']
        row['positive_negative_proportion'] = REVIEWS_DICT[review_id]['positive_negative_proportion']
        row['negative_positive_proportion'] = REVIEWS_DICT[review_id]['negative_positive_proportion']
        row['negative_score'] = REVIEWS_DICT[review_id]['negative_score']
        row['positive_score'] = REVIEWS_DICT[review_id]['positive_score']
        row['neutral_score'] = REVIEWS_DICT[review_id]['neutral_score']
        row['compound_score'] = REVIEWS_DICT[review_id]['compound_score']
        row['review_score'] = REVIEWS_DICT[review_id]['review_score']
        row['user_earned_more'] = row['user_points'] >= row['bot_points']
        row['bot_earned_more'] = row['user_points'] < row['bot_points']
        row['review_prob'] = PROBA2GO_DICT[str(review_id)]
        row['current_game_mistakes_percentage'] = row['current_game_mistakes_amount'] / row['rounds_so_far'] if row['rounds_so_far'] != 0 else 0
        row['total_games_mistakes_percentage'] = row['total_games_mistakes_amount'] / row['total_rounds_so_far'] if row['total_rounds_so_far'] != 0 else 0
        row['last_reaction_time_bins_categories'] = row['last_reaction_time']
        row['lost_cause'] = lost_cause(row, self.bot_thresholds)
        return row


    def _baseline(self, row, user_noise, **kwargs):
        """Generates a baseline reaction time (returns -1)."""
        return -1
    
    def _random(self, row, user_noise, **kwargs):
        """Generates a random reaction time using specified distribution."""
        sampling_distribution = kwargs.get('rt_sampling_distribution', 'normal')
        if sampling_distribution == 'normal':
            result = np.random.normal(5000, 10000)
            while result < 0:
                result = np.random.normal(5000, 10000)
            return result
        elif sampling_distribution == 'uniform':
            return np.random.uniform(0, 30000)
        else:
            raise ValueError(f'Unexpected distribution: {sampling_distribution}')


    def _model(self, row, user_noise, **kwargs):
        """Generates a reaction time using a trained model."""
        if self.trained_model is None:
            raise ValueError("Model is not loaded")
        X_row = [row[feature] for feature in self.model_features]
        return self.trained_model.predict([X_row])[0]

    def _heuristic(self, row, user_noise, **kwargs):
        """Generates a reaction time using heuristic calculations."""
        neutral_sampling = kwargs.get('rt_neutral_sampling', 'normal')
        frustration_std_method = kwargs.get('rt_frustration_std_method', '+')
        rt_baseline_std = kwargs.get('rt_baseline_std', 0)
        w_word_count = kwargs.get('rt_w_word_count', 150)

        user_strategy = row['user_strategy_name']
        word_count = row['word_count']
        neutral_score = row['neutral_score']
        current_game_mistakes_percentage = row['current_game_mistakes_percentage']
        last_didWin_True = int(row['last_didWin_True'])
        last_last_didWin_True = int(row['last_last_didWin_True'])
        lost_cause = row['lost_cause']
        game_id = row['gameId']
        round_num = row['roundNum']

        baseline = self.strategies_rt_baselines[user_strategy]
        if user_strategy != 'random':
            baseline = np.random.normal(baseline, rt_baseline_std)
        if frustration_std_method == "+":
            w_frustration = abs(np.random.normal(game_id*round_num, game_id+round_num))
        else:
            w_frustration = abs(np.random.normal(game_id*round_num, game_id/round_num))

        if neutral_sampling == 'normal':
            w_neutral = abs(np.random.normal(600, 100))
        else:
            w_neutral = int(neutral_sampling)

        w_lost_cause = np.random.uniform(0.2, 0.5)
        if user_strategy == 'random':
            w_strategy = np.random.uniform(0, 0.5)
            rt = baseline + (1 - lost_cause)*(word_count*w_word_count*w_strategy + neutral_score*w_neutral*w_strategy) - lost_cause*w_lost_cause*baseline
        elif user_strategy == "correct - oracle":
            w_strategy = np.random.uniform(0.5, 1)
            rt = baseline + word_count*w_word_count*w_strategy - lost_cause*w_lost_cause*baseline - w_frustration - user_noise
        else: # LLM/Trustful
            w_strategy = np.random.uniform(0.5, 1)
            w_mistakes = abs(np.random.normal(1000, 200))
            rt = baseline + word_count*w_word_count*w_strategy - lost_cause*w_lost_cause*baseline + \
            lost_cause*w_lost_cause*neutral_score*w_neutral*w_strategy + (1-lost_cause)*neutral_score*w_neutral*w_strategy - \
            (1-lost_cause)*current_game_mistakes_percentage*w_mistakes + (1-lost_cause)*(last_didWin_True+last_last_didWin_True)*100 - w_frustration - user_noise
        rt = max(rt, 300)
        return rt

    def generate_rt(self, row, user_noise):
        """Generates a reaction time based on the selected method."""
        if self.method != 'random' or self.method != 'baseline':
            row = self.extract_features(row)
        return self.generator_func(row, user_noise, **self.kwargs)