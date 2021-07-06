import numpy as np
from collections import defaultdict
import argparse

from browser_tools import Browser
from tetrio_tools import get_online_game_sp, get_playername_sp
from solver_tools import Solver, SolverStats, SolverConfig
from render_tools import ReplayVideo, RendererConfig
# import render_tools


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", "-C", action="store_true", help="disable loading replays from cache")
parser.add_argument("--seed", "-S", default=None, type=int, help="random seed for randomly chosen games")
parser.add_argument("--speedup", "-s", default=5, type=int, help="speedup when recording replays")
parser.add_argument("--record", "-r", action="store_true", help="save rendered replay videos for each player")
parser.add_argument("--force", "-f", action="store_true", help="ignore errors in reading frames")
parser.add_argument("--verbose", "-v", action="store_true", help="display all debugging information")
parser.add_argument("targets", type=str, nargs="+", help="replay ID (with r:) or link, path to local replay, or '?' for random")


def main(args):
    browser = Browser(speedup=args.speedup, verbose=args.verbose)
    SolverConfig.set_verbose(args.verbose).set_force(args.force)
    RendererConfig.set_verbose(args.verbose).set_scale(20)
    rng = np.random.RandomState(args.seed)

    # Get all replays
    fulldata = defaultdict(list)
    for target in args.targets:
        if target.startswith("?"):
            # info, data = get_random_game(browser, rng, use_cache=not args.no_cache)
            raise NotImplementedError("Random singleplayer games not yet supported")
        elif "r:" in target:
            info, data = get_online_game_sp(browser, target.split("r:")[-1], use_cache=not args.no_cache)
        else:
            # info, data = get_custom_game(browser, target, use_cache=not args.no_cache)
            raise NotImplementedError("Uploaded singleplayer games not yet supported")
        fulldata[get_playername_sp(info)].append(data)
    browser.finish()
    print("Retrieved all replay data")

    zipd = lambda x: [dict(zip(x.keys(), vs)) for vs in zip(*(x[k] for k in x.keys()))]
    todict = lambda f: {
        "frame": f["grid"]["frame"],
        "grid": f["grid"]["value"],
        "next": f["next"]["value"],
        "hold": f["hold"]["value"]
    }
    metrics = {
        "_ipieces": (lambda x: x["event"].piece == "I"),
        "_iattack": (lambda x: x["event"].piece == "I" and x["A"]["total_attack"]),
        "Single": (lambda x: not x["A"]["tspin"] and x["A"]["lines"] == 1),
        "Double": (lambda x: not x["A"]["tspin"] and x["A"]["lines"] == 2),
        "Triple": (lambda x: not x["A"]["tspin"] and x["A"]["lines"] == 3),
        "Tetris": (lambda x: x["A"]["tetris"]),
        "_tpieces": (lambda x: x["event"].piece == "T"),
        "T Spin Single": (lambda x: x["A"]["tspin"] and x["A"]["lines"] == 1),
        "T Spin Double": (lambda x: x["A"]["tspin"] and x["A"]["lines"] == 2),
        "T Spin Triple": (lambda x: x["A"]["tspin"] and x["A"]["lines"] == 3),
        "Perfect Clear": (lambda x: x["A"]["perfect_clear"]),

        "Tetris Attack": (lambda x: x["A"]["tetris"] and x["A"]["attack"]["base"]),
        "T Spin Attack": (lambda x: x["A"]["tspin"] and x["A"]["attack"]["base"]),
        "Combo Attack": (lambda x: x["A"]["attack"]["combo"]),
        "B2B Bonus": (lambda x: x["A"]["attack"]["b2b"]),
        "_vscounter": (lambda x: x["A"]["attack"]["total_attack"] + x["A"]["garbage"])
    }
    for player, data in fulldata.items():
        video = ReplayVideo()
        stats = {
            "_frames": 0,
            "Time": 0,
            "Pieces": 0,
            "Attack": 0,
            "PPS": 0,
            "APM": 0,
            "VS": 0,
            "Atk/Bag": 0,
            "T Efficiency": 0,
            "I Attack": 0,
            "Max Combo": 0,
            "Max B2B": 0,
            "Max Spike": 0,
            **{k:0 for k in metrics}
        }
        overhangs = defaultdict(int)
        placements = defaultdict(int)
        for rnum, rounddata in enumerate(data):
            filtered = list(frame for frame in zipd(rounddata) if frame["next"]["value"] != "-----")
            assert all(f["grid"]["frame"] == f["next"]["frame"] == f["hold"]["frame"] for f in filtered)
            # Analyze
            print(f"Round {rnum + 1}")
            slvr = Solver()
            piecelist = slvr.compute_piece_list(list(map(todict, filtered)))
            reconstruction = list(slvr.reconstruct(map(todict, filtered), piecelist))
            events = list(slvr.compute_attacks(reconstruction))
            # events = events[7 * 3:]
            # if not events: continue
            # print(sum(ev["A"]["total_attack"] for ev in events))
            SolverStats.tspin_overhangs(events, out=overhangs)
            SolverStats.tspin_columns(events, out=placements)
            # Update stats
            stats["Pieces"] += len(events)
            stats["_frames"] += filtered[-1]["grid"]["frame"] # - events[0]["frame"]
            stats["Attack"] += sum(ev["A"]["total_attack"] for ev in events)
            stats["Max Combo"] = max(stats["Max Combo"], max(ev["A"]["combo"] for ev in events))
            stats["Max B2B"] = max(stats["Max B2B"], max(ev["A"]["b2b"] for ev in events))
            stats["Max Spike"] = max(stats["Max Spike"], max(ev["A"]["spike"] for ev in events))
            for k in metrics:
                stats[k] += sum(map(metrics[k], events))
            # Render video
            if args.record:
                video.render(*events)
                video.extend_by(120)
        stats["Time"] = stats["_frames"] / 60
        stats["PPS"] = stats["Pieces"] * 60 / stats["_frames"]
        stats["APM"] = stats["Attack"] * 60 * 60 / stats["_frames"]
        stats["VS"] = stats["_vscounter"] * 100 * 60 / stats["_frames"]
        stats["Atk/Bag"] = stats["Attack"] / stats["Pieces"] * 7
        stats["T Efficiency"] = (stats["T Spin Single"] + 2*stats["T Spin Double"] + 3*stats["T Spin Triple"]) / (2 * stats["_tpieces"])
        stats["I Attack"] = stats["_iattack"] / stats["_ipieces"]

        # Save video
        if args.record:
            video.save(f"videos/{player}_replays.mp4")

        print(f"Player: {player}")
        for k in stats:
            if not k.startswith("_"):
                print(f"{k}: {stats[k]}")
        print("T Spin Overhangs")
        print(" ".join(f"{k: >3}" for k in "SZTLJIO"))
        print(" ".join(f"{overhangs[k]: >3}" for k in "SZTLJIO"))
        print("T Spin Columns")
        print("  ".join(f"{placements[k]}" for k in range(10)))
        print()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)