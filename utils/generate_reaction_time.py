from consts import PROBA2GO_DICT, REVIEWS_DICT
import warnings
import pickle
import os
import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class ReactionTimeGenerator:
    def __init__(self, method, rt_model_file_name = None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.trained_model = None
        self.model_features = None
        self.method = method
        if rt_model_file_name:
            with open(os.path.join('models', rt_model_file_name), 'rb') as file:
                self.trained_model = pickle.load(file)
                self.model_features = self.trained_model.feature_names_in_

        self._set_generator_function()

    def _set_generator_function(self):
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
        row['review_length'] = REVIEWS_DICT[review_id]['total_len']
        row['positive_review_proportion'] = REVIEWS_DICT[review_id]['positive_proportion']
        row['negative_review_proportion'] = REVIEWS_DICT[review_id]['negative_proportion']
        row['positive_negative_proportion'] = REVIEWS_DICT[review_id]['positive_negative_proportion']
        row['negative_positive_proportion'] = REVIEWS_DICT[review_id]['negative_positive_proportion']
        row['user_earned_more'] = row['user_points'] >= row['bot_points']
        row['bot_earned_more'] = row['user_points'] < row['bot_points']
        row['review_prob'] = PROBA2GO_DICT[str(review_id)]
        row['current_game_mistakes_percentage'] = row['current_game_mistakes_amount'] / row['rounds_so_far'] if row['rounds_so_far'] != 0 else 0
        row['total_games_mistakes_percentage'] = row['total_games_mistakes_amount'] / row['total_rounds_so_far'] if row['total_rounds_so_far'] != 0 else 0
        row['last_reaction_time_bins_categories'] = row['last_reaction_time']
        return row


    def _baseline(self, row, *args, **kwargs):
        return -1
    
    def _random(self, row, sampling_distribution = 'normal', *args, **kwargs):
        if sampling_distribution == 'normal':
            result = np.random.normal(5000, 10000)
            while result < 0:
                result = np.random.normal(5000, 10000)
            return result
        elif sampling_distribution == 'uniform':
            return np.random.uniform(0, 30000)
        else:
            raise ValueError(f'Unexpected distribution: {sampling_distribution}')


    def _model(self, row, *args, **kwargs):
        if self.trained_model is None:
            raise ValueError("Model is not loaded")
        X_row = [row[feature] for feature in self.model_features]
        return self.trained_model.predict([X_row])[0]

    def _heuristic(self, row, *args, **kwargs):
        pass

    def generate_rt(self, row):
        if self.method != 'random' or self.method != 'baseline':
            row = self.extract_features(row)
        return self.generator_func(row, *self.args, **self.kwargs)


# def generate_reaction_time(row):
#     user_strategy_name = row['user_strategy_name']
#     row['played_oracle'] = 0
#     row['played_random'] = 0
#     row['played_trustful'] = 0
#     row['played_llm'] = 0
#     if user_strategy_name == 'history_and_review_quality (Trustful)':
#         row['played_llm'] = 1
#     elif user_strategy_name == 'random':
#         row['played_random'] = 1
#     elif user_strategy_name == 'correct - oracle':
#         row['played_oracle'] = 1
#     elif user_strategy_name == 'LLM stochastic (Language-based)':
#         row['played_llm'] = 1
#     else:
#         raise ValueError(f'Unexpected strategy: {user_strategy_name}')
    
#     review_id = row['reviewId']
#     row['review_length'] = REVIEWS_DICT[review_id]['total_len']
#     row['positive_review_proportion'] = REVIEWS_DICT[review_id]['positive_proportion']
#     row['negative_review_proportion'] = REVIEWS_DICT[review_id]['negative_proportion']
#     row['positive_negative_proportion'] = REVIEWS_DICT[review_id]['positive_negative_proportion']
#     row['negative_positive_proportion'] = REVIEWS_DICT[review_id]['negative_positive_proportion']
#     row['user_earned_more'] = row['user_points'] >= row['bot_points']
#     row['bot_earned_more'] = row['user_points'] < row['bot_points']
#     row['review_prob'] = PROBA2GO_DICT[str(review_id)]
#     row['current_game_mistakes_percentage'] = row['current_game_mistakes_amount'] / row['rounds_so_far'] if row['rounds_so_far'] != 0 else 0
#     row['total_games_mistakes_percentage'] = row['total_games_mistakes_amount'] / row['total_rounds_so_far'] if row['total_rounds_so_far'] != 0 else 0
#     row['last_reaction_time_bins_categories'] = row['last_reaction_time']
#     X_row = [[row[feature] for feature in RT_MODEL_FEATURES]]
#     reaction_time = RT_MODEL.predict(X_row)[0]
#     return reaction_time
