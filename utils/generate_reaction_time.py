from consts import RT_MODEL, PROBA2GO_DICT, REVIEWS_DICT, RT_MODEL_FEATURES
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def generate_reaction_time(row):
    review_id = row['reviewId']
    row['review_length'] = REVIEWS_DICT[review_id]['total_len']
    row['positive_review_proportion'] = REVIEWS_DICT[review_id]['positive_proportion']
    row['user_earned_more'] = row['user_points'] >= row['bot_points']
    row['bot_earned_more'] = row['user_points'] < row['bot_points']
    row['review_prob'] = PROBA2GO_DICT[str(review_id)]
    row['current_game_mistakes_percentage'] = row['current_game_mistakes_amount'] / row['rounds_so_far'] if row['rounds_so_far'] != 0 else 0
    row['total_games_mistakes_percentage'] = row['total_games_mistakes_amount'] / row['total_rounds_so_far'] if row['total_rounds_so_far'] != 0 else 0
    row['last_reaction_time_bins_categories'] = row['last_reaction_time']
    X_row = [[row[feature] for feature in RT_MODEL_FEATURES]]
    reaction_time = RT_MODEL.predict(X_row)[0]
    return reaction_time
