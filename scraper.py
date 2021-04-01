# %%
import pandas as pd
from src import pg_engine, scraper


def get_scrape_results(scrape_pbp, season):
    if(scrape_pbp):
        return scraper.get_scrape_results_seasons([season])
    else:
        games = pg_engine.query_sql(f"""
            Select * from games where season LIKE '{season}%'""")

        plays = pg_engine.query_sql(f"""
            Select p.* from plays p 
            Inner join games g ON p.game_id = g.id
            Where g.season LIKE '{season}%'
            order by g.id, p.idx""")

        return {
            'games': games,
            'plays': plays
        }


seasons = [2010, 2011, 2012, 2013, 2014,
           2015, 2016, 2017, 2018, 2019]

seasons = ["20102011", "20112012", "20122013", "20132014", "20142015", "20152016", ""]

for season in seasons:
    scrape_results = get_scrape_results(False, season)
    plays = scrape_results['plays']
    games = scrape_results['games']

    # pg_engine.insert_batched(scrape_results['games'], 'games')
    # pg_engine.insert_batched(scrape_results['plays'], 'plays')

    # if('players' in scrape_results):
    #     pg_engine.insert_batched(scrape_results['players'], 'players')

    # game_strength_states = scraper.generate_game_strength_states(plays, games)
    # pg_engine.insert_batched(game_strength_states, 'game_strength_states')

    # team_games = scraper.generate_team_games(games)
    # pg_engine.insert_batched(team_games, 'team_games')

    shot_plays = scraper.generate_shot_plays(plays, games)
    pg_engine.insert_batched(shot_plays, 'player_play_shots')
# %%

seasons = ["20162017"]
game_strength_states = pd.DataFrame()

for season in seasons:
    print("""
        Select p.* from plays p 
        Inner join games g ON p.game_id = g.id
        Where g.season = '%s'
        order by g.id, p.idx        """ % (season))

    plays = pg_engine.query_sql("""
        Select p.* from plays p 
        Inner join games g ON p.game_id = g.id
        Where g.season = '%s'
        order by g.id, p.idx
        """ % (season))

    games = pg_engine.query_sql("""
        Select * from games where season = '%s'
        """ % (season))

    game_strength_states = game_strength_states.append(
        scraper.generate_game_strength_states(plays, games))

# %%
pg_engine.insert_batched(game_strength_states, 'game_strength_states')
# %%

seasons = ["20162017", "20172018", "20182019", "20192020"]
team_games = pd.DataFrame()

for season in seasons:
    games = pg_engine.query_sql("""
        Select * from games where season = '%s'
        """ % (season))

    team_games = team_games.append(scraper.generate_team_games(games))

# %%
pg_engine.insert_batched(team_games, 'team_games')
# %%
team_games
# %%

seasons = ["20162017", "20172018", "20182019", "20192020"]
shot_plays = pd.DataFrame()

for season in seasons:
    plays = pg_engine.query_sql("""
        Select p.* from plays p 
        Inner join games g ON p.game_id = g.id
        Where g.season = '%s'
        order by g.id, p.idx
        """ % (season))

    games = pg_engine.query_sql("""
        Select * from games where season = '%s'
        """ % (season))

    shot_plays = shot_plays.append(scraper.generate_shot_plays(plays, games))

# %%
pg_engine.insert_batched(shot_plays, 'player_play_shots')
# %%
shot_plays

# %%
import hockey_scraper

scrape_results = hockey_scraper.scrape_games([2017020001], False, data_format='Pandas', docs_dir=True)
# %%
