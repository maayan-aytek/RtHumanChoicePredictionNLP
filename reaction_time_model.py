import os
import json
import pickle
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from Simulation.dm_strategies import BOT_ACTION, REVIEWS
from consts import DATA_CLEAN_ACTION_PATH_X, DATA_CLEAN_ACTION_PATH_Y, reaction_time_bins, PROBA2GO_DICT, REVIEWS_DICT

hotel_dfs = dict()
with open("data/baseline_proba2go.txt", 'r') as file:
    probs_dict = json.load(file)

def add_review_prob(row):
    review_id = row['reviewId']
    return probs_dict[str(review_id)]

def get_review_scores(row):
    review_id = row['reviewId']
    review_scores_dict = REVIEWS_DICT[review_id]
    return review_scores_dict


def add_mistakes_columns(df):
    # Calculate the cumulative sum of mistakes, then shift within each group so the calculation will be correct to the start of the round
    df['mistakes_cumulative'] = df.groupby(['user_id', 'gameId'])['didWin'].transform(lambda x: (~x).cumsum())  
    df['current_game_mistakes_amount'] = df.groupby(['user_id', 'gameId'])['mistakes_cumulative'].shift(fill_value=0)  
    
    # Calculate cumulative mistakes across all games for each user, then shift  
    df['total_mistakes_cumulative'] = df.groupby(['user_id'])['didWin'].transform(lambda x: (~x).cumsum())  
    df['total_games_mistakes_amount'] = df.groupby(['user_id'])['total_mistakes_cumulative'].shift(fill_value=0)  
    
    # For percentage calculations, adjust for the shift by avoiding division by zero  
    df['rounds_so_far'] = df.groupby(['user_id', 'gameId']).cumcount()  
    df['total_rounds_so_far'] = df.groupby(['user_id']).cumcount() 
    df['current_game_mistakes_percentage'] = df['current_game_mistakes_amount'] / df['rounds_so_far'].replace(0, pd.NA)  
    df['total_games_mistakes_percentage'] = df['total_games_mistakes_amount'] / df['total_rounds_so_far'].replace(0, pd.NA)  
    
    # Fill NaN values in the percentage columns  
    df.fillna({'current_game_mistakes_percentage': 0, 'total_games_mistakes_percentage': 0}, inplace=True)  
    df.drop(['mistakes_cumulative', 'total_mistakes_cumulative', 'rounds_so_far', 'total_rounds_so_far'], axis=1, inplace=True)
    return df


def history_and_review_quality(history_window, quality_threshold, information):      
    if len(information["previous_rounds"]) == 0 \
            or history_window == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS] >= 8)
                                    or (r[BOT_ACTION] <= 8 and r[REVIEWS] < 8)) for r in
                                information["previous_rounds"][
                                -history_window:]])) == 1:          
        if information["bot_message"] >= quality_threshold: 
            return 1
        else:
            return 0
    else:
        return 0
    

def LLM_based(information):
    proba2go = {int(k): v for k, v in PROBA2GO_DICT.items()}
    review_llm_score = proba2go[information["review_id"]]
    return int(np.random.rand() <= review_llm_score)


def create_information(group):
    group['information'] = [
        [(hotel_score, review_score) for hotel_score, review_score in zip(group['hotelScore'][:i], group['review_score'][:i])] 
        for i in range(len(group))
    ]
    return group


def calculate_played_oracle(group):
    did_win = group['didWin'].values
    played_oracle = [0] * len(did_win)
    for i in range(1, len(did_win) - 1):
        if all(did_win[i: len(group)]):
            played_oracle[i:] = [1] * (len(group) - i)
            break
    group['played_oracle'] = played_oracle
    return group


def calculate_trustful_and_llm(row, user_properties):
    user_id = row['user_id']
    review_id = row['reviewId']
    bot_message = row['review_score']
    history_window = user_properties[user_id]['history_window']
    quality_threshold = user_properties[user_id]['quality_threshold']
    information = {'previous_rounds': row['information'], 'bot_message': bot_message, 'review_id': review_id}
    
    trustful_decision = history_and_review_quality(history_window, quality_threshold, information)
    llm_decision = LLM_based(information)
    
    return pd.Series([trustful_decision, llm_decision])


def lost_cause(row, strategy_threshold_dict):
    if row['strategy_id'] not in strategy_threshold_dict:
        threshold = 9
    else:
        threshold = strategy_threshold_dict[row['strategy_id']]
    if 10 - threshold < row['current_game_mistakes_amount']:
        return 1 
    else:
        return 0


def pre_process(actions_df, bot_thresholds):
    bin_edges = [-1] + [b[0] for b in reaction_time_bins] + [reaction_time_bins[-1][1]]
    actions_df['reaction_time_bins'] = pd.cut(actions_df['reaction_time'], bins=bin_edges, include_lowest=True)
    actions_df['last_reaction_time_bins'] = pd.cut(actions_df['last_reaction_time'], bins=bin_edges, include_lowest=True)
    actions_df['last_reaction_time_bins_categories'] = actions_df['last_reaction_time_bins'].apply(lambda x: (x.left))
    actions_df['reaction_time_bins_categories'] = actions_df['reaction_time_bins'].apply(lambda x: (x.left))
    scores_series = actions_df.apply(get_review_scores, axis=1)
    scores_df = pd.DataFrame(scores_series.tolist())
    actions_df = pd.concat([actions_df, scores_df], axis=1)
    actions_df = add_mistakes_columns(actions_df)
    actions_df['user_earned_more'] = actions_df['user_points'] >= actions_df['bot_points']
    actions_df['bot_earned_more'] = actions_df['user_points'] < actions_df['bot_points']
    actions_df['review_prob'] = actions_df.apply(add_review_prob, axis=1)

    user_properties = {
    user_id: {
        'history_window': np.random.negative_binomial(2, 1 / 2) + np.random.randint(0, 2),
        'quality_threshold': np.random.normal(8, 0.5),
        'random_noise': np.random.normal(0, 0.1)
    } 
    for user_id in actions_df['user_id'].unique()
    }

    strategies_df = actions_df[['user_id', 'gameId', 'reviewId', 'strategy_id', 'hotelScore', 'review_score', 'didWin', 'didGo']] \
        .groupby(by=['user_id', 'gameId']).apply(create_information).reset_index(drop=True)
    strategies_df = strategies_df.groupby(by=['user_id', 'gameId']).apply(calculate_played_oracle).reset_index(drop=True)
    strategies_df[['trustful_decision', 'llm_decision']] = strategies_df.apply(lambda row: calculate_trustful_and_llm(row, user_properties), axis=1)
    strategies_df['played_trustful'] = (strategies_df['trustful_decision'] == strategies_df['didGo']).astype(int)
    strategies_df['played_llm'] = (strategies_df['llm_decision'] == strategies_df['didGo']).astype(int)
    actions_df = actions_df.merge(strategies_df)
    actions_df['played_random'] = actions_df.apply(lost_cause, args=(bot_thresholds,), axis=1)
    actions_df.loc[actions_df[actions_df['played_oracle'] == 1].index, ['played_trustful', 'played_llm', 'played_random']] = 0
    return actions_df


def run(seed, min_samples_leaf, class_weight, n_estimators=20, top_features='all'):
    model_name = f'min_samples_leaf_{min_samples_leaf}_class_weight_{class_weight}_top_features_{top_features}_seed_{seed}.pkl'
    if os.path.exists(f'sweep_models/{model_name}'):
        return model_name
    X_df = pd.read_csv(DATA_CLEAN_ACTION_PATH_X)
    bot_thresholds_X = {3: 10, 0: 7, 2: 9, 5: 9, 59: 8, 19: 9}
    train_df = pre_process(X_df, bot_thresholds_X)

    all_features = ['gameId', 'roundNum', 'user_points', 'bot_points', 'last_didGo_True', 'last_last_didGo_True', 'last_didWin_True', 'last_last_didWin_True', 'review_prob', 'review_score',
                'review_length', 'positive_review_proportion','negative_review_proportion','positive_negative_proportion','negative_positive_proportion', 'negative_score',
                'positive_score', 'neutral_score', 'compound_score',
                'current_game_mistakes_amount', 'current_game_mistakes_percentage', 'total_games_mistakes_percentage', 'total_games_mistakes_amount',
                'user_earned_more', 'bot_earned_more', 'last_reaction_time_bins_categories', 'played_trustful', 'played_llm', 'played_random', 'played_oracle']
    classification_label_column = 'reaction_time_bins_categories'
    X_train, y_train =  train_df[all_features], train_df[classification_label_column]
    classification_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight=class_weight)
    classification_model.fit(X_train[all_features], y_train)
    importances_df = pd.DataFrame({"feature_names" : classification_model.feature_names_in_, "importances" : classification_model.feature_importances_}).sort_values(by='importances', ascending=False)
    if top_features != 'all': 
        chosen_features = importances_df[:int(top_features)]['feature_names'].tolist()
    else:
        chosen_features = all_features
    
    final_classification_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight=class_weight)
    final_classification_model.fit(X_train[chosen_features], y_train)
    with open(f'sweep_models/{model_name}', 'wb') as f:
        pickle.dump(final_classification_model, f)

    return model_name


