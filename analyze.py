import numpy as np
from collections import defaultdict
import argparse

from browser_tools import Browser
from tetrio_tools import get_full_game, get_random_game, get_custom_game, get_playername
from solver_tools import Solver, SolverStats, SolverConfig
from new_solver import PieceSolver, BoardSolver
# import render_tools


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", "-C", action="store_true", help="disable loading replays from cache")
parser.add_argument("--seed", "-S", default=None, type=int, help="random seed for randomly chosen games")
parser.add_argument("--manual", "-m", action="store_true", help="manually drag config file into Tetr.IO (faster)")
parser.add_argument("--speedup", "-s", default=2, type=int, help="speedup when recording replays")
parser.add_argument("--verbose", "-v", action="store_true", help="display all debugging information")
parser.add_argument("targets", type=str, nargs="+", help="replay ID (with r:) or link, path to local replay, or '?' for random")


def main(args):
    browser = Browser(speedup=args.speedup, manual_config=args.manual, verbose=args.verbose)
    SolverConfig.set_verbose(args.verbose)
    rng = np.random.RandomState(args.seed)

    # Get all replays
    fulldata = defaultdict(list)
    for target in args.targets:
        if target.startswith("?"):
            info, data = get_random_game(browser, rng, use_cache=not args.no_cache)
        elif "r:" in target:
            info, data = get_full_game(browser, target.split("r:")[-1], use_cache=not args.no_cache)
        else:
            info, data = get_custom_game(browser, target, use_cache=not args.no_cache)
        for player in (0, 1):
            fulldata[get_playername(info, player)].extend(rdata[player] for rdata in data)
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
    for player, data in fulldata.items():
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
            "Max Spike": 0,
            **{k:0 for k in metrics}
        }
        overhangs = defaultdict(int)
        placements = defaultdict(int)
        for rounddata in data:
            filtered = list(frame for frame in zipd(rounddata) if frame["next"]["value"] != "-----")
            assert all(f["grid"]["frame"] == f["next"]["frame"] == f["hold"]["frame"] for f in filtered)
            # Analyze
            print("Next")
            ps, bs = PieceSolver(), BoardSolver()
            nfs = list(map(todict, filtered))
            nfs = ps.compute(nfs)
            nfs = bs.compute(nfs)
            continue
            slvr = Solver()
            newfs = list(slvr.calc_events(map(todict, filtered)))
            # slvr.find_interesting_parts(newfs)
            SolverStats.tspin_overhangs(newfs, out=overhangs)
            SolverStats.tspin_columns(newfs, out=placements)
            # Update stats
            stats["Pieces"] += sum(f["nplaced"] for f in newfs[1:])
            stats["_frames"] += newfs[-1]["end"]
            stats["Attack"] += sum(f["attack"] for f in newfs)
            stats["Max Combo"] = max(stats["Max Combo"], max(f["combo"] for f in newfs[1:]))
            stats["Max B2B"] = max(stats["Max B2B"], max(f["b2b"] for f in newfs[1:]))
            stats["Max Spike"] = max(stats["Max Spike"], max(f["attack_combo"] for f in newfs[1:]))
            for k in metrics:
                stats[k] += sum(map(metrics[k], newfs[1:]))
        raise KeyboardInterrupt()
        stats["Time"] = stats["_frames"] / 60
        stats["PPS"] = stats["Pieces"] * 60 / stats["_frames"]
        stats["APM"] = stats["Attack"] * 60 * 60 / stats["_frames"]
        stats["Atk/Bag"] = stats["Attack"] / stats["Pieces"] * 7
        stats["T Efficiency"] = (stats["T Spin Single"] + 2*stats["T Spin Double"] + 3*stats["T Spin Triple"]) / (2 * stats["_tpieces"])
        stats["I Attack"] = stats["_iattack"] / stats["_ipieces"]

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