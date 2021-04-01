import requests
import json
import pandas as pd
from tqdm.notebook import tqdm


def get_shots_seasons(url, first, seasons, strength, player=None):
    df = pd.DataFrame()

    for season in seasons:
        print(f'Grabbing data for {season}')

        if player is None:
            df_season, total, has_next_page, cursor = get_shots(
                url, None, first, season, strength)
        else:
            df_season, total, has_next_page, cursor = get_player_shots(
                player, url, None, first, season, strength)

        pbar = tqdm(total=total)

        while has_next_page is True:
            if player is None:
                df_new, total, has_next_page, cursor = get_shots(
                    url, cursor, first, season, strength)
            else:
                df_new, total, has_next_page, cursor = get_player_shots(
                    player, url, cursor, first, season, strength)

            df_season = df_season.append(df_new)

            pbar.update(len(df_new))

        pbar.close()
        df = df.append(df_season)

    return df


def get_shots(url, cursor, first, season, strength):
    # print(get_query(cursor, first, season, strength))

    response = requests.post(
        url, json={'query': get_query(cursor, first, season, strength)})
    json_data = json.loads(response.text)

    return get_shots_from_season_data(json_data['data']['seasons'][0])


def get_player_shots(player, url, cursor, first, season, strength):
    response = requests.post(url, json={'query': get_player_query(
        player, cursor, first, season, strength)})
    json_data = json.loads(response.text)

    return get_shots_from_season_data(json_data['data']['players'][0]['seasons'][0])


def get_shots_from_season_data(season_data):
    total = season_data['shots']['fenwickEvents']['total']
    has_next_page = season_data['shots']['fenwickEvents']['pageInfo']['hasNextPage']
    cursor = season_data['shots']['fenwickEvents']['pageInfo']['endCursor']
    df_data = season_data['shots']['fenwickEvents']['edges']
    df_data = list(map(lambda edge: edge['node'], df_data))
    df = pd.DataFrame(df_data)

    return df, total, has_next_page, cursor


def get_query(cursor, first, season, strength):
    cursor_filter = "(after: \"{}\", first: {})".format(
        cursor, first) if cursor is not None else ""
    strength_filter = "strength: {}".format(
        strength) if strength is not None else ""
    shots_filter = "(filter: {%s})" % strength_filter if strength is not None else ""
    season_filter = "(filter: {id: \"%s\" })" % season

    return """
    {
      seasons%s {
        shots%s {
          fenwickEvents%s {
            total
            pageInfo {
              startCursor
              endCursor
              hasNextPage
            }
            edges {
              node {
                player
                type
                period
                homeGame
                coordX
                coordY
                angle
                distance
                shotType
                event
                priorEvent
                team
                priorEventTeam
                angleChange
                distanceChange
                leading
                leadingByOne
                trailing
                trailingByOne
                secondsChange
              }
            }
          }
        }
      }
    }
    """ % (season_filter, shots_filter, cursor_filter)


def get_player_query(player, cursor, first, season, strength):
    player_filter = "(filter: {name: \"%s\" })" % player
    query = get_query(cursor, first, season, strength)

    return """
    {
        players%s
        %s
    }
    """ % (player_filter, query)
