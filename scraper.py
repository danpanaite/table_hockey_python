# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import psycopg2

import math
import pandas as pd
import hockey_scraper
from tqdm import tqdm
from sqlalchemy import create_engine

engine = create_engine(
    'postgres://begehrbqtxewcj:cfb27a4ff83dbcc3fef1d7a8e40fa176587a9ab71fadea07aa1c945f94c68fda@ec2-52-200-48-116.compute-1.amazonaws.com:5432/de2bcjaimtiij')

data = engine.execute("Select * from players limit 10").fetchall()

print(data)


# %%

# pd.options.display.float_format = '{:.5f}'.format

plays = pd.DataFrame()
# scrape_results = hockey_scraper.scrape_seasons(
#     [2015, 2016, 2018, 2019], False, data_format='Pandas', docs_dir=True)
scrape_results = hockey_scraper.scrape_games(
    [2017020001], False, data_format='Pandas')

# %%
pbp = scrape_results['pbp']
# pbp = scrape_results_test['pbp']

plays['game_id'] = pbp['Game_Id_Json'].astype(int)
plays['idx'] = pbp.index
plays['id'] = (plays['game_id'].astype(str) +
               plays['idx'].astype(str)).astype(float)
plays['period'] = pbp['Period']
plays['game_seconds'] = pbp['Seconds_Elapsed'].astype(int)
plays['type'] = pbp['Event']
plays['zone'] = pbp['Ev_Zone']
plays['detail'] = pbp['Type']
plays['team_abv'] = pbp['Ev_Team']
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

# %%
scrape_results['players'].to_sql(
    'game_players', con=engine, if_exists="append", index=False, method='multi')

# %%
scrape_results['games'].to_sql(
    'games', con=engine, if_exists="append", index=False, method='multi')
# %%

i = 0
increment = 5000
total = len(plays)
pbar = tqdm(total=len(plays))

while i <= total:
    plays[i:i+increment].to_sql('plays', con=engine,
                                if_exists="append", index=False, method='multi')
    i += increment
    pbar.update(increment)

pbar.close()

# %%

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
    shot_plays['game'] = shot_plays.apply(
        lambda row: games[games['id'] == row['game_id']], axis=1)

    print('test')
    shot_plays['home_game'] = shot_plays.apply(
        lambda row: row['game']['home_team'] == row['team_abv'], axis=1)

    shot_plays['opponent_team'] = shot_plays.apply(
        lambda row: row['game']['away_team'] if row['home_game'] else row['game']['home_team'], axis=1)

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

    shot_plays_team['season'] = shot_plays.apply(
        lambda row: row['game']['season'], axis=1)

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


player_play_shots = generate_shot_plays(plays, scrape_results['games'])

# %%
i = 0
increment = 5000
total = len(plays)
pbar = tqdm(total=len(plays))

while i <= total:
    player_play_shots[i:i+increment].to_sql('player_play_shots', con=engine,
                                            if_exists="append", index=False, method='multi')
    i += increment
    pbar.update(increment)

pbar.close()
# %%
