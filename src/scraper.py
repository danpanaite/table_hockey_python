import math
import pandas as pd
import hockey_scraper
from tqdm import tqdm
from sqlalchemy import create_engine


team_play_types = {
    'SHOT': 'SHOTONGOAL',
    'GOAL': 'GOAL',
    'MISS': 'SHOTMISSED',
    'BLOCK': 'SHOTBLOCKED'
}

opponent_play_types = {
    'SHOT': 'SAVE',
    'GOAL': 'GOALAGAINST',
    'MISS': 'SHOTMISSEDAGAINST',
    'BLOCK': 'BLOCKED'
}

adjusted_team_abvs = {
    'L.A': 'LAK',
    'S.J': 'SJS',
    'T.B': 'TBL',
    '#': 'BUF',
    'N.J': 'NJD'
}


def get_scrape_results_seasons(seasons):
    scrape_results = hockey_scraper.scrape_seasons(
        seasons, False, data_format='Pandas', docs_dir='/workspaces/table_hockey_python/hockey_scraper_data')

    return {
        'plays': get_plays_from_pbp(scrape_results['pbp']),
        'games': scrape_results['games'],
        'players': scrape_results['players']
    }


def get_scrape_results_games(game_ids):
    scrape_results = hockey_scraper.scrape_games(
        game_ids, False, data_format='Pandas', docs_dir='/workspaces/table_hockey_python/hockey_scraper_data')

    return {
        'plays': get_plays_from_pbp(scrape_results['pbp']),
        'games': scrape_results['game'],
        'players': scrape_results['players']
    }


def get_plays_from_pbp(pbp):
    plays = pd.DataFrame()
    plays['game_id'] = pbp['Game_Id_Json'].astype(int)
    plays['idx'] = pbp.index
    plays['id'] = (plays['game_id'].astype(str) +
                   plays['idx'].astype(str)).astype(float)
    plays['period'] = pbp['Period']
    plays['game_seconds'] = pbp['Seconds_Elapsed'].astype(int)
    plays['type'] = pbp['Event']
    plays['zone'] = pbp['Ev_Zone']
    plays['detail'] = pbp['Type']
    plays['team_abv'] = pbp.apply(lambda row: adjusted_team_abvs[row['Ev_Team']]
                                  if row['Ev_Team'] in adjusted_team_abvs else row['Ev_Team'], axis=1)
    plays['player_1'] = pbp['p1_ID'].astype('Int64')
    plays['player_2'] = pbp['p2_ID'].astype('Int64')
    plays['player_3'] = pbp['p3_ID'].astype('Int64')
    plays['coords_x'] = pbp['xC'].astype('Int64')
    plays['coords_y'] = pbp['yC'].astype('Int64')
    plays['home_score'] = pbp['Home_Score'].astype(int)
    plays['away_score'] = pbp['Away_Score'].astype(int)
    plays['home_skaters'] = pbp['Home_Players'].astype(int)
    plays['away_skaters'] = pbp['Away_Players'].astype(int)
    plays['game_strength_state'] = pbp['Strength'].str.replace('x', 'v')
    plays['home_goalie'] = pbp['Home_Goalie_Id'].astype('Int64')
    plays['away_goalie'] = pbp['Away_Goalie_Id'].astype('Int64')

    plays['distance'] = pbp['Description'].str.extract(
        r'(\d{1,}(?= ft))').astype(float)

    return plays


def generate_shot_plays(plays, games):
    print('Generating shot plays')
    print('Recording last play')
    plays['last_play_type'] = plays.shift(1)['type']
    plays['last_play_team'] = plays.shift(1)['team_abv']
    plays['last_play_coords_x'] = plays.shift(1)['coords_x']
    plays['last_play_coords_y'] = plays.shift(1)['coords_y']
    plays['seconds_change'] = plays['game_seconds'] - \
        plays.shift(1)['game_seconds']

    print('Building shot plays')
    shot_plays = plays[plays['type'].isin(['SHOT', 'GOAL', 'MISS', 'BLOCK'])]
    shot_plays = pd.merge(shot_plays, games, left_on='game_id', right_on='id')
    shot_plays['type'] = shot_plays['type_x']
    shot_plays['home_score'] = shot_plays['home_score_x']
    shot_plays['away_score'] = shot_plays['away_score_x']
    shot_plays['home_game'] = shot_plays['home_team'] == shot_plays['team_abv']
    shot_plays['opponent_team'] = shot_plays.apply(
        lambda row: row['away_team'] if row['home_game'] else row['home_team'], axis=1)

    print('goal_coords_x')
    shot_plays['goal_coords_x'] = shot_plays.apply(
        lambda row: get_goal_coords_x(row), axis=1)

    print('Calculating shot distance')
    shot_plays['distance'] = shot_plays.apply(
        lambda row: math.sqrt((row['coords_x'] - row['goal_coords_x'])**2) + row['coords_y']**2, axis=1)

    print('Calculating shot angle')
    shot_plays['angle'] = shot_plays.apply(
        lambda row: math.asin(row['coords_y'] / row['distance']) * 180 / math.pi if row['distance'] > 0 else 0, axis=1)

    shot_plays['last_play_distance'] = shot_plays.apply(
        lambda row: math.sqrt((row['last_play_coords_x'] - row['goal_coords_x'])**2) + row['last_play_coords_y']**2, axis=1)

    shot_plays['last_play_angle'] = shot_plays.apply(
        lambda row: math.asin(row['last_play_coords_y'] / row['last_play_distance']) * 180 / math.pi if row['last_play_distance'] > 0 else 0, axis=1)

    shot_plays['distance_change'] = shot_plays.apply(
        lambda row: math.sqrt(
            (row['coords_x'] - row['last_play_coords_x'])**2 +
            (row['coords_y'] - row['last_play_coords_y'])**2), axis=1)

    shot_plays['angle_change'] = shot_plays.apply(
        lambda row: abs(row['angle'] - row['last_play_angle']), axis=1)

    shot_plays['angle'] = shot_plays.apply(
        lambda row: abs(row['angle']), axis=1)

    return pd.concat([get_shot_plays_team(shot_plays, is_team_play=True),
                      get_shot_plays_team(shot_plays, is_team_play=False)])

def get_shot_plays_team(shot_plays, is_team_play):
    print('Building shot plays for team')

    shot_plays_team = pd.DataFrame()
    team_key = 'team_abv' if is_team_play else 'opponent_team'
    opponent_team_key = 'opponent_team' if is_team_play else 'team_abv'
    play_types = team_play_types if is_team_play else opponent_play_types

    shot_plays_team['id'] = shot_plays['game_id'].astype(
        str) + shot_plays['idx'].astype(str) + shot_plays[team_key]

    shot_plays_team['type'] = shot_plays.apply(
        lambda row: play_types[row['type']], axis=1)

    shot_plays_team['season'] = shot_plays['season']

    shot_plays_team['team_season_id'] = shot_plays[team_key] + \
        shot_plays_team['season'].astype(str)

    shot_plays_team['team_game_id'] = shot_plays[team_key] + \
        shot_plays['game_id'].astype(str)

    shot_plays_team['player'] = shot_plays.apply(
        lambda row: get_player(row, is_team_play), axis=1)

    shot_plays_team['opponent'] = shot_plays.apply(
        lambda row: get_opponent(row, is_team_play), axis=1)

    shot_plays_team['team_season_player_id'] = shot_plays[team_key] + \
        shot_plays_team['season'].astype(
        str) + shot_plays_team['player'].astype(str)

    shot_plays_team['team'] = shot_plays['team_abv']
    shot_plays_team['home_game'] = shot_plays['home_game']
    shot_plays_team['opponent_team'] = shot_plays[opponent_team_key]

    shot_plays_team['period'] = shot_plays['period']
    shot_plays_team['game_seconds'] = shot_plays['game_seconds']
    shot_plays_team['strength_state'] = shot_plays.apply(
        lambda row: row['game_strength_state']
        if row['home_game'] else get_reversed_state(row['game_strength_state']), axis=1)

    shot_plays_team['trailing'] = shot_plays.apply(
        lambda row: row['home_score'] < row['away_score']
        if row['home_game'] else row['away_score'] < row['home_score'], axis=1)

    shot_plays_team['trailing_by_one'] = shot_plays.apply(
        lambda row: row['home_score'] - row['away_score'] == -1
        if row['home_game'] else row['away_score'] - row['home_score'] == -1, axis=1)

    shot_plays_team['leading'] = shot_plays.apply(
        lambda row: row['home_score'] > row['away_score']
        if row['home_game'] else row['away_score'] > row['home_score'], axis=1)

    shot_plays_team['leading_by_one'] = shot_plays.apply(
        lambda row: row['home_score'] - row['away_score'] == 1
        if row['home_game'] else row['away_score'] - row['home_score'] == 1, axis=1)

    shot_plays_team['shot_type'] = shot_plays['detail']
    shot_plays_team['coords_x'] = shot_plays['coords_x']
    shot_plays_team['coords_y'] = shot_plays['coords_y']
    shot_plays_team['distance'] = shot_plays['distance']
    shot_plays_team['angle'] = shot_plays['angle']
    shot_plays_team['distance_change'] = shot_plays['distance_change']
    shot_plays_team['angle_change'] = shot_plays['angle_change']
    shot_plays_team['seconds_change'] = shot_plays['seconds_change']
    shot_plays_team['last_play_type'] = shot_plays['last_play_type']
    shot_plays_team['last_play_team'] = shot_plays['last_play_team']

    return shot_plays_team


def get_goal_coords_x(shot):
    goal_coords_x = 89 if shot['coords_x'] > 0 else -89

    if (shot['distance'] is not None and
        shot['distance'] > 89 and
        shot['detail'] != 'Tip-In' and
        shot['detail'] != 'Wrap-around' and
        shot['detail'] != 'Deflected' and
            not (shot['distance'] > 89 and shot['zone'] == "OFF")):
        goal_coords_x = -89 if shot['coords_x'] > 0 else 89

    return goal_coords_x


def get_player(play, is_team_play):
    if is_team_play:
        return play['player_1'] if play['type'] != 'BLOCK' else play['player_2']
    elif play['type'] == 'BLOCK':
        return play['player_1']
    else:
        return play['away_goalie'] if play['home_game'] else play['home_goalie']


def get_opponent(play, is_team_play):
    if is_team_play:
        if play['type'] == 'BLOCK':
            return play['player_1']
        else:
            return play['away_goalie'] if play['home_game'] else play['home_goalie']

    return play['player_1'] if play['type'] != 'BLOCK' else play['player_2']


def get_shot_play_type(play, team_play):
    if(team_play):
        return team_play_types[play['type']]
    else:
        return opponent_play_types[play['type']]


def get_reversed_state(state):
    return f'{state[2]}v{state[0]}'


def generate_game_strength_states(plays, games):
    game_strength_states = pd.DataFrame()
    start_period = 1

    game_seconds = 0
    start_index = 0
    start_seconds = 0
    game_strength_state = '5v5'

    for _, game in tqdm(games.iterrows(), total=len(games)):
        game_plays = plays[plays['game_id'] == game['id']]

        for index, (_, play) in enumerate(game_plays.iterrows()):
            if index > 0:
                seconds_elapsed = game_plays.iloc[index]['game_seconds'] - \
                    game_plays.iloc[index - 1]['game_seconds']
                game_seconds = game_seconds + seconds_elapsed if seconds_elapsed > 0 else game_seconds

            if play['game_strength_state'] == game_strength_state:
                continue

            duration = game_seconds - start_seconds

            game_strength_states = game_strength_states.append([{
                'id': '%s%s%s' % (play['game_id'], str(start_index), game['home_team']),
                'game_id': play['game_id'],
                'period': start_period,
                'team_season_id': '%s%s' % (game['home_team'], game['season']),
                'team': game['home_team'],
                'strength_state': game_strength_state,
                'start_idx': start_index,
                'end_idx': play['idx'],
                'duration': duration
            }])

            game_strength_states = game_strength_states.append([{
                'id': '%s%s%s' % (play['game_id'], str(start_index), game['away_team']),
                'game_id': play['game_id'],
                'period': start_period,
                'team_season_id': '%s%s' % (game['away_team'], game['season']),
                'team': game['away_team'],
                'strength_state': get_reversed_state(game_strength_state),
                'start_idx': start_index,
                'end_idx': play['idx'],
                'duration': duration
            }])

            start_period = play['period']
            start_seconds = game_seconds
            start_index = play['idx']
            game_strength_state = play['game_strength_state']

    return game_strength_states


def generate_team_games(games):
    team_games = pd.DataFrame()

    for _, game in tqdm(games.iterrows(), total=len(games)):

        team_games = team_games.append([{
            'id': '%s%s' % (game['home_team'], game['id']),
            'game_id': game['id'],
            # 'team_season_id': '%s%s' % (game['home_team'], game['season']),
            'season': game['season'],
            'team': game['home_team'],
            'opponent': game['away_team'],
            'team_score': game['home_score'],
            'opponent_score': game['away_score'],
            'home_game': True,
            'game_end': game['game_end'],
            'points': get_points(game['home_score'], game['away_score'], game['game_end']),
            'rest_days': get_rest_days(game['home_team'], game, games),
        }])

        team_games = team_games.append([{
            'id': '%s%s' % (game['away_team'], game['id']),
            'game_id': game['id'],
            # 'team_season_id': '%s%s' % (game['away_team'], game['season']),
            'season': game['season'],
            'team': game['away_team'],
            'opponent': game['home_team'],
            'team_score': game['away_score'],
            'opponent_score': game['home_score'],
            'home_game': False,
            'game_end': game['game_end'],
            'points': get_points(game['away_score'], game['home_score'], game['game_end']),
            'rest_days': get_rest_days(game['away_team'], game, games),
        }])

    return team_games


def get_points(team_score, opponent_score, game_end):
    return 2 if team_score > opponent_score else 1 if opponent_score > team_score and (game_end == 'OT' or game_end == 'SO') else 0


def get_rest_days(team, game, games):
    previous_games = games[((games['home_team'] == team) | (games['away_team'] == team))
                           & (games['start_time'] < game['start_time'])]

    if(previous_games.empty):
        return 0

    previous_game = previous_games.sort_values(
        by='start_time', ascending=False).iloc[0]

    return (game['start_time'] - previous_game['start_time']).days
