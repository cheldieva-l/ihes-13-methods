"""Bounded full-state Korf/IDA* pilot; never uses subword replacement.

Input: original 54-label states and a validated baseline for fallback only.
Output: independent solver paths, failure reasons and a fully replay-checked CSV.
An unfinished search is retryable, not an invalid puzzle or optimality proof.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time

from compact_adapter import CompactAdapter, EXE, HERE, from_twsearch, native_replay, read_source_states


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_baseline(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["initial_state_id"]): row["path"].strip().split(".") if row["path"].strip() else [] for row in rows}
    return result if len(rows) == len(result) else None


def valid(adapter, state, word):
    try:
        return native_replay(state, word, adapter.moves) == adapter.central
    except (KeyError, IndexError, TypeError):
        return False


def solve_original(adapter, pid, state, output, cache, seconds, depth, memory, threads):
    """Solve one source state directly; return candidates and bounded-search evidence."""
    compact, reason = adapter.encode(state)
    if reason:
        return dict(puzzle_id=pid, status="encoding_error", reason=reason, candidates=[])
    if state == adapter.central:
        return dict(puzzle_id=pid, status="solved", candidates=[[]], seconds=0, timed_out=False)
    scratch = output / f"puzzle_{pid}.scr"
    scratch.write_text(adapter.scramble_text(f"puzzle_{pid}", compact), encoding="utf-8")
    command = [str(EXE), "-q", "-M", str(memory), "-t", str(threads),
               "--cachedir", str(cache), "--startprunedepth", "6", "--fillpref", "0",
               "--maxdepth", str(depth), str(HERE / "ordinary333_compact_exact18.tws"), str(scratch)]
    started = time.monotonic()
    tail = collections.deque(maxlen=35)
    candidates = []
    process = subprocess.Popen(command, cwd=cache, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def drain():
        with (output / f"puzzle_{pid}.log").open("w", encoding="utf-8") as log:
            for raw in process.stdout:
                log.write(raw)
                line = raw.strip()
                tail.append(line)
                parts = line.split()
                word = [from_twsearch(token) for token in parts]
                if parts and all(token is not None for token in word):
                    candidates.append(word)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=max(0.01, seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        process.wait(timeout=15)
    reader.join(timeout=10)
    return dict(puzzle_id=pid, status="timeout_retryable" if timed_out else "search_returned" if process.returncode == 0 else "solver_error",
                candidates=candidates, command=command, timed_out=timed_out, exit_code=process.returncode,
                seconds=round(time.monotonic()-started, 3), tail=list(tail))


def run(args):
    """Run original-state pilot and retain independently checked fallbacks on failure."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "pruning_cache"
    cache.mkdir(exist_ok=True)
    adapter = CompactAdapter()
    states, source_errors = read_source_states()
    paths = read_baseline(args.input)
    invalid = [] if paths is None else [pid for pid, state in states.items() if pid not in paths or not valid(adapter, state, paths[pid])]
    if paths is None or set(paths) != set(states) or invalid or source_errors or adapter.errors:
        save_json(output / "status.json", dict(status="invalid_input", invalid_ids=invalid,
                                              source_errors=source_errors, adapter_errors=adapter.errors))
        return
    seed_score = sum(map(len, paths.values()))
    original = {pid: len(word) for pid, word in paths.items()}
    rows = []
    started = time.monotonic()
    deadline = started + args.total_seconds
    header = dict(method="official Rokicki twsearch: full-state Korf-style iterative deepening with solved-state hashed pruning",
                  subword_search=False, centers_normalized=False, proof_of_global_optimum=False,
                  seed_score=seed_score, started_utc=stamp(), threads=args.threads, memory_mib=args.memory)
    for pid in args.ids:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (output / "STOP").exists():
            break
        if pid not in states:
            rows.append(dict(puzzle_id=pid, status="unknown_id"))
            continue
        save_json(output / "status.json", dict(header, status="running", puzzle_id=pid, updated_utc=stamp(), rows=rows))
        # The incumbent is only a depth ceiling and fallback; no route word is sent to twsearch.
        result = solve_original(adapter, pid, states[pid], output, cache, min(args.seconds, remaining),
                                max(0, original[pid]), args.memory, args.threads)
        checks = [dict(path=".".join(word), length=len(word), valid=valid(adapter, states[pid], word))
                  for word in result.pop("candidates")]
        good = [row for row in checks if row["valid"]]
        best = min(good, key=lambda row: row["length"]) if good else None
        if best and best["length"] < len(paths[pid]):
            paths[pid] = best["path"].split(".") if best["path"] else []
        row = dict(result, returned_paths=checks, before=original[pid], after=len(paths[pid]),
                   full_state_solution_found=bool(good), updated_utc=stamp())
        rows.append(row)
        save_json(output / f"puzzle_{pid}.json", row)
        print(json.dumps({k:row[k] for k in ("puzzle_id","status","before","after","full_state_solution_found")}), flush=True)
    invalid = [pid for pid, state in states.items() if not valid(adapter, state, paths[pid])]
    score = sum(map(len, paths.values()))
    final = output / f"submission_{score}.csv"
    if not invalid:
        with final.open("w", encoding="utf-8", newline="") as stream:
            writer=csv.writer(stream,quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["initial_state_id", "path"])
            writer.writerows((pid,".".join(paths[pid]) or " ") for pid in sorted(paths))
    solved_rows = [r for r in rows if r.get("full_state_solution_found")]
    summary = dict(header, status="pilot_complete" if len(rows)==len(args.ids) else "budget_or_stop",
                   updated_utc=stamp(), rows=rows, elapsed_seconds=round(time.monotonic()-started, 2),
                   independent_solved_count=len(solved_rows), source_count=len(states),
                   standalone_full_dataset_score=None, fallback_inclusive_score=score,
                   moves_saved=seed_score-score, invalid_export_ids=invalid,
                   submission=str(final) if not invalid else None)
    save_json(output / "status.json", summary)
    print(json.dumps({k:summary[k] for k in ("status","independent_solved_count","fallback_inclusive_score","moves_saved")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=HERE/"pilot_20260914")
    parser.add_argument("--ids",type=int,nargs="+",default=[9,20,49,50,950,999])
    parser.add_argument("--seconds",type=float,default=30)
    parser.add_argument("--total-seconds",type=float,default=200)
    parser.add_argument("--memory",type=int,default=512)
    parser.add_argument("--threads",type=int,default=2)
    run(parser.parse_args())


