from typing import AnyStr, Dict, List
import requests
import pickle
import json
import sys
import os
from browser_tools import Browser


# Download stats functions

def download_players():
    url = "https://ch.tetr.io/api/users/lists/league?limit=100"
    r = requests.get(url)
    return r.json()["data"]["users"]

def download_games(userid: AnyStr):
    url = f"https://ch.tetr.io/api/streams/league_userrecent_{userid}"
    r = requests.get(url)
    return r.json()["data"]["records"]

def download_replay(replayid: AnyStr):
    url = f"https://tetr.io/api/games/{replayid}"
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZmNmMGFmYmFjOGNlNDIxYmU3OGMyNjciLCJpYXQiOjE2MTE5NjE2ODN9.Bquojb4icUzPq0L8JO4GR5Tt9Nfx9pgCY-Ie7Q6WpMY",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "pragma": "no-cache",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin"
    }
    r = requests.get(url, headers=headers)
    return r.json()

def upload_replay(filename: AnyStr):
    with open(filename) as f:
        return {"game": json.load(f)}

# Extract info functions

def get_userid(player: Dict):
    return player["_id"]

def get_replayid(record: Dict):
    return record["replayid"]

def n_games(replaydata: Dict):
    return len(replaydata["game"]["data"])

def n_frames(replaydata: Dict, roundnum: int):
    return min(p["frames"] for p in replaydata["game"]["data"][roundnum]["replays"])

def get_playername(replaydata: Dict, idx: int):
    return replaydata["game"]["endcontext"][idx]["user"]["username"]


def get_random_game(browser: Browser, rng, **kwargs):
    # Select a player
    players = download_players()
    i = rng.randint(len(players))
    player = players[i]
    print(f"Choosing player: '{player['username']}' with {player['league']['rating']:.0f} TR")
    # Select a game
    games = download_games(get_userid(player))
    i = rng.randint(len(games))
    game = games[i]
    players = [entry['user']['username'] for entry in game['endcontext']]
    opponent = [name for name in players if name != player['username']][0]
    print(f"\tversus opponent: '{opponent}'")
    replayid = get_replayid(game)
    return get_full_game(browser, replayid, **kwargs)


def get_full_game(browser: Browser, replayid: AnyStr, maxrounds=15, use_cache=True):
    print(f"Replay ID:\tr:{replayid}")
    if use_cache:
        try:
            with open(f"cache/replay_{replayid}.pkl", "rb") as f:
                return download_replay(replayid), pickle.load(f)
        except FileNotFoundError:
            print("Replay not cached, capturing replay")
    replays = download_replay(replayid)
    framecounts = [n_frames(replays, idx) - 10 for idx in range(n_games(replays))]
    data = list(browser.get(replayid, range(min(n_games(replays), maxrounds)), framecounts))
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/replay_{replayid}.pkl", "wb") as f:
        pickle.dump(data, f)
        print("Saved replay to cache")
    return replays, data


def get_custom_game(browser: Browser, filename: AnyStr, maxrounds=15, use_cache=True):
    sname = filename.split("\\")[-1].split("/")[-1].replace(".ttrm", "")
    print(f"Replay Name:\t{sname}")
    if use_cache:
        try:
            with open(f"cache/replay_{sname}.pkl", "rb") as f:
                return upload_replay(filename), pickle.load(f)
        except FileNotFoundError:
            print("Replay not cached, capturing replay")
    replays = upload_replay(filename)
    framecounts = [n_frames(replays, idx) - 10 for idx in range(n_games(replays))]
    data = list(browser.get(filename, range(min(n_games(replays), maxrounds)), framecounts))
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/replay_{sname}.pkl", "wb") as f:
        pickle.dump(data, f)
        print("Saved replay to cache")
    return replays, data
