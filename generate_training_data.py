import numpy as np
from collections import defaultdict
import argparse
import csv

from browser_tools import Browser
from tetrio_tools import get_online_game_sp, download_playerinfo, download_records, get_userid, get_replayid
from solver_tools import Solver, SolverStats, SolverConfig

# Output format (CSV):
# width,height,board(flat),piece,x(left),y(top),rotation

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", "-C", action="store_true", help="disable loading replays from cache")
parser.add_argument("--speedup", "-s", default=1, type=int, help="speedup when recording replays")
parser.add_argument("--force", "-f", action="store_true", help="ignore errors in reading frames")
parser.add_argument("--verbose", "-v", action="store_true", help="display all debugging information")
parser.add_argument("targets", type=str, nargs="+", help="username(s) of player(s) to gather data from")
parser.add_argument("output", type=str, help="csv file to output data to")


def main(args):
    browser = Browser(speedup=args.speedup, verbose=args.verbose)
    SolverConfig.set_verbose(args.verbose).set_force(args.force)

    # Get all records
    replayids = []
    for target in args.targets:
        userid = get_userid(download_playerinfo(target))
        records = map(get_replayid, download_records(userid, "40l"))
        replayids.extend(records)
    print("Retrieved replay IDs")
    # Get all replays
    fulldata = []
    for replayid in replayids:
        _, data = get_online_game_sp(browser, replayid, use_cache=not args.no_cache)
        fulldata.append(data)
    browser.finish()
    print("Retrieved all replay data")

    zipd = lambda x: [dict(zip(x.keys(), vs)) for vs in zip(*(x[k] for k in x.keys()))]
    todict = lambda f: {
        "frame": f["grid"]["frame"],
        "grid": f["grid"]["value"],
        "next": f["next"]["value"],
        "hold": f["hold"]["value"]
    }

    with open(args.output, "w") as out:
        writer = csv.writer(out)
        for rnum, rounddata in enumerate(fulldata):
            filtered = list(frame for frame in zipd(rounddata) if frame["next"]["value"] != "-----")
            assert all(f["grid"]["frame"] == f["next"]["frame"] == f["hold"]["frame"] for f in filtered)
            # Analyze
            print(f"Replay {rnum + 1}")
            slvr = Solver()
            try:
                piecelist = slvr.compute_piece_list(list(map(todict, filtered)))
            except AssertionError as err:
                if args.force:
                    continue
                else:
                    raise err
            reconstruction = list(slvr.reconstruct(map(todict, filtered), piecelist))
            events = list(slvr.compute_attacks(reconstruction))
            softdrops = defaultdict(int)
            SolverStats.soft_drops(events, out=softdrops)
            softdrops = sum(softdrops.values())
            print(f"Run contained {softdrops} soft drop(s)")
            print()
            for ev in events:
                height, width = ev["board"].shape
                flatboard = "".join(cell for row in ev["board"] for cell in row)
                writer.writerow((width, height, flatboard, ev["event"].piece, ev["event"].x, ev["event"].y, ev["event"].rotation))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)