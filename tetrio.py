from typing import AnyStr, Dict, List
import requests
import pickle
import json
import numpy as np
from collections import defaultdict
import cv2
import sys
import argparse

import browser
import solver
import render


# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", "-C", action="store_true", help="disable loading replays from cache")
parser.add_argument("--seed", "-S", default=0, type=int, help="random seed for randomly chosen games")
parser.add_argument("--manual", "-m", action="store_true", help="manually drag config file into Tetr.IO (faster)")
parser.add_argument("--speedup", "-s", default=2, type=int, help="speedup when recording replays")
parser.add_argument("target", type=str, nargs="*", help="replay ID (with r:) or link, or path to local replay")

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


def random_game(bws, seed=None, **kwargs):
    rng = np.random.RandomState(seed)
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
    return full_game(bws, replayid, **kwargs)


def full_game(bws, replayid, maxrounds=15, use_cache=True):
    print(f"Replay ID:\tr:{replayid}")
    if use_cache:
        try:
            with open(f"cache/replay_{replayid}.pkl", "rb") as f:
                return download_replay(replayid), pickle.load(f)
        except FileNotFoundError:
            print("Replay not cached, capturing replay")
    replays = download_replay(replayid)
    framecounts = [n_frames(replays, idx) - 10 for idx in range(n_games(replays))]
    data = list(bws.get(replayid, range(min(n_games(replays), maxrounds)), framecounts))
    # page = browser.open_tetrio()
    # browser.open_replay(page, replayid)
    # data = []
    # for roundnum in range(min(n_games(replays), maxrounds)):
    #     framecount = n_frames(replays, roundnum) - 10 # As a safety measure
    #     print(f"Recording round {roundnum + 1}")
    #     browser.capture_replay(page, roundnum, framecount)
    #     data.append(browser.all_replay_data(page))
    # page.close()
    with open(f"cache/replay_{replayid}.pkl", "wb") as f:
        pickle.dump(data, f)
        print("Saved replay to cache")
    return replays, data


def custom_game(bws, filename, maxrounds=15, use_cache=True):
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
    data = list(bws.get(None, range(min(n_games(replays), maxrounds)), framecounts))
    # page = browser.open_tetrio()
    # print("Drag your replay file into the browser...")
    # browser.open_replay(page)
    # data = []
    # for roundnum in range(min(n_games(replays), maxrounds)):
    #     framecount = n_frames(replays, roundnum) - 10 # As a safety measure
    #     print(f"Recording round {roundnum + 1}")
    #     browser.capture_replay(page, roundnum, framecount)
    #     data.append(browser.all_replay_data(page))
    # page.close()
    with open(f"cache/replay_{sname}.pkl", "wb") as f:
        pickle.dump(data, f)
        print("Saved replay to cache")
    return replays, data

# Unused

def retrieve_pieces_list(data):
    pieces = ""
    for frame in data:
        nextps = frame["next"]
        if nextps == "-----": continue
        for k in range(5, -1, -1):
            if pieces.endswith(nextps[:k]):
                pieces = pieces + nextps[k:]
                break
    pieces = list(set("TSZIOLJ") - set(pieces[:6]))[0] + pieces
    if not all(set("TSZIOLJ") == set(pieces[i:i + 7]) for i in range(0, len(pieces) - 7, 7)):
        print("Found invalid piece sequence", "".join(pieces), sep="\n", file=sys.stderr)
    return pieces

def nextpiece_by_frame(data):
    pieces = retrieve_pieces_list(data)
    frames = []
    lastindex = 0
    for f in data:
        if all(p == "-" for p in f["next"]): continue
        while f["frame"] > len(frames):
            frames.append(frames[-1])
        index = pieces[lastindex:].index(f["next"]) + lastindex - 1
        frames.append(pieces[index + 1:index + 6])
        lastindex = index
    return frames

def not_blank(frame):
    return frame["next"]["value"] != "-----"

def count_errors(data):
    return sum(1 for rd in data for f in rd[0]["next"] if f["value"] == "-----")


def main(args):
    bws = browser.Browser(speedup=args.speedup, manual_config=args.manual)
    if not args.target:
        info, data = random_game(bws, seed=args.seed, use_cache=not args.no_cache)
    elif "r:" in args.target[0]:
        info, data = full_game(bws, args.target[0].split("r:")[-1], use_cache=not args.no_cache)
    else:
        info, data = custom_game(bws, args.target[0], use_cache=not args.no_cache)
    print("Retrieved all game data")
    print(f"Missed {count_errors(data)} frames", file=sys.stderr)

    zipd = lambda x: [dict(zip(x.keys(), vs)) for vs in zip(*(x[k] for k in x.keys()))]
    todict = lambda f: {
        "frame": f["grid"]["frame"],
        "grid": f["grid"]["value"],
        "next": f["next"]["value"],
        "hold": f["hold"]["value"]
    }
    fullvideos = []
    metrics = {
        "_ipieces": (lambda x: (x["piece_placed"] or {}).get("piece", "") == "I"),
        "_iattack": (lambda x: metrics["_ipieces"](x) and x["attack"]),
        "Single": (lambda x: not x["tspin"] and x["cleared"] == 1),
        "Double": (lambda x: not x["tspin"] and x["cleared"] == 2),
        "Triple": (lambda x: not x["tspin"] and x["cleared"] == 3),
        "Tetris": (lambda x: x["cleared"] == 4),
        "_tpieces": (lambda x: (x["piece_placed"] or {}).get("piece", "") == "T"),
        "T Spin Single": (lambda x: x["tspin"] and x["cleared"] == 1),
        "T Spin Double": (lambda x: x["tspin"] and x["cleared"] == 2),
        "T Spin Triple": (lambda x: x["tspin"] and x["cleared"] == 3),
        "Perfect Clear": (lambda x: x["perfect_clear"]),

        "Tetris Attack": (lambda x: x["atk"]["base"] if x["cleared"] == 4 else 0),
        "T Spin Attack": (lambda x: x["atk"]["base"] if x["tspin"] else 0),
        "Combo Attack": (lambda x: x["atk"]["combo"] if x["cleared"] else 0),
        "B2B Bonus": (lambda x: x["atk"]["b2b"] if x["cleared"] else 0)
    }
    for player in [0, 1]:
        stats = {
            "_frames": 0,
            "Time": 0,
            "Pieces": 0,
            "Attack": 0,
            "PPS": 0,
            "APM": 0,
            "Atk/Bag": 0,
            "T Efficiency": 0,
            "I Attack": 0,
            "Max Combo": 0,
            "Max B2B": 0,
            "Max Atk Combo": 0,
            **{k:0 for k in metrics}
        }
        videos = []
        overhangs = defaultdict(int)
        placements = defaultdict(int)
        for rounddata in data:
            filtered = list(filter(not_blank, zipd(rounddata[player])))
            assert all(f["grid"]["frame"] == f["next"]["frame"] == f["hold"]["frame"] for f in filtered)
            # Count largest gap
            print(
                "Dropped at most",
                max(fB["grid"]["frame"] - fA["grid"]["frame"] for fA, fB in zip(filtered, filtered[1:])),
                "frames", file=sys.stderr
            )
            # Analyze
            slvr = solver.Solver()
            newfs = list(slvr.calc_events(map(todict, filtered)))
            # slvr.find_interesting_parts(newfs)
            solver.SolverTools.tspin_overhangs(newfs, out=overhangs)
            solver.SolverTools.tspin_columns(newfs, out=placements)
            # Update stats
            stats["Pieces"] += sum(f["nplaced"] for f in newfs[1:])
            stats["_frames"] += newfs[-1]["end"]
            stats["Attack"] += sum(f["attack"] for f in newfs)
            stats["Max Combo"] = max(stats["Max Combo"], max(f["combo"] for f in newfs[1:]))
            stats["Max B2B"] = max(stats["Max B2B"], max(f["b2b"] for f in newfs[1:]))
            stats["Max Atk Combo"] = max(stats["Max Atk Combo"], max(f["attack_combo"] for f in newfs[1:]))
            for k in metrics:
                # stats[k] += sum(1 for _ in filter(metrics[k], newfs[1:]))
                stats[k] += sum(map(metrics[k], newfs[1:]))
            # Render video
            video = render.create_video(newfs, nextpiece_by_frame(newfs), interpolation=False, length=filtered[-1]["grid"]["frame"])
            # for frame in slvr.log:
            #     render.display_image(video[frame])
            # render.display_video(video)
            videos.append(video)
            videos.append(np.broadcast_to(video[-1], (120,) + video.shape[1:]))
        stats["Time"] = stats["_frames"] / 60
        stats["PPS"] = stats["Pieces"] * 60 / stats["_frames"]
        stats["APM"] = stats["Attack"] * 60 * 60 / stats["_frames"]
        stats["Atk/Bag"] = stats["Attack"] / stats["Pieces"] * 7
        stats["T Efficiency"] = (stats["T Spin Single"] + 2*stats["T Spin Double"] + 3*stats["T Spin Triple"]) / (2 * stats["_tpieces"])
        # stats["I Efficiency"] = (0.25*stats["_idouble"] + 0.5*stats["_itriple"] + stats["Tetris"]) / (0.7 * stats["_ipieces"])
        stats["I Attack"] = stats["_iattack"] / stats["_ipieces"]

        print(f"Player: {get_playername(info, player)}")
        for k in stats:
            if not k.startswith("_"):
                print(f"{k}: {stats[k]}")
        # print(solver.SolverTools.visualize_placements(placements))
        print("T Spin Overhangs")
        print(" ".join(f"{k: >3}" for k in "SZTLJIO"))
        print(" ".join(f"{overhangs[k]: >3}" for k in "SZTLJIO"))
        print("T Spin Columns")
        print("  ".join(f"{placements[k]}" for k in range(10)))
        print()
        video = np.concatenate(videos)
        fullvideos.append(video)
    # video = render.side_by_side(fullvideos)
    # render.display_video(video)
    # render.save_video("full_game.mp4", video)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)