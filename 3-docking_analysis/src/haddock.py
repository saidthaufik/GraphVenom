#!/usr/bin/env python3
"""Run prepared HADDOCK3 pair folders in parallel with a live terminal dashboard."""

from __future__ import annotations

# Publication layout bootstrap; model equations and training parameters are unchanged.
from pathlib import Path as _PublicationPath
import sys as _publication_sys
_PUB_ROOT = next(p for p in _PublicationPath(__file__).resolve().parents if (p / ".rki-publication.json").is_file())
_publication_sys.path.insert(0, str(_PUB_ROOT / "2-machine_learning_analysis"))

import argparse
import csv
import signal
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    from rich import box
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = _PUB_ROOT / "3-docking_analysis" / "cache" / "haddock_pairs_manifest.csv"
DEFAULT_STATUS = _PUB_ROOT / "3-docking_analysis" / "cache" / "haddock_batch_status.csv"
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="haddock_pairs_manifest.csv from prepare_haddock_inputs.py.")
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS, help="Batch status CSV output.")
    parser.add_argument("--max-parallel", type=int, default=2, help="Number of HADDOCK jobs to run concurrently.")
    parser.add_argument("--datasets", default="ab51,ab66", help="Comma-separated datasets to run.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N pending jobs after filtering.")
    parser.add_argument("--rank-from", type=int, default=None, help="Minimum rank to run.")
    parser.add_argument("--rank-to", type=int, default=None, help="Maximum rank to run.")
    parser.add_argument("--force", action="store_true", help="Rerun jobs even when final capri_ss.tsv exists.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned jobs without running HADDOCK.")
    parser.add_argument("--haddock-bin", default="haddock3", help="HADDOCK3 executable.")
    parser.add_argument("--poll-sec", type=float, default=5.0, help="Progress polling interval in seconds.")
    parser.add_argument("--no-live", action="store_true", help="Disable rich live dashboard and use plain logs.")
    return parser.parse_args()


def handle_signal(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[STOP] Received signal {signum}; no new jobs will be started.", flush=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def final_score_file(pair_dir: Path) -> Path:
    return pair_dir / "run_peptide_score" / "analysis" / "6_caprieval_analysis" / "capri_ss.tsv"


def is_done(row: dict[str, str]) -> bool:
    return final_score_file(Path(row["pair_dir"])).exists()


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    datasets = {dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()}
    out: list[dict[str, str]] = []
    for row in rows:
        rank = int(row["rank"])
        if row["dataset"] not in datasets:
            continue
        if args.rank_from is not None and rank < args.rank_from:
            continue
        if args.rank_to is not None and rank > args.rank_to:
            continue
        if is_done(row) and not args.force:
            continue
        out.append(row)
    out.sort(key=lambda row: (row["dataset"], int(row["rank"])))
    if args.limit is not None:
        out = out[: args.limit]
    return out


def parse_best_score(score_file: Path) -> dict[str, str]:
    if not score_file.exists():
        return {}
    with score_file.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        try:
            row = next(reader)
        except StopIteration:
            return {}
    return {
        "best_model": row.get("model", ""),
        "haddock_score": row.get("score", ""),
        "vdw": row.get("vdw", ""),
        "elec": row.get("elec", ""),
        "desolv": row.get("desolv", ""),
        "bsa": row.get("bsa", ""),
    }


def read_existing_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def write_status(path: Path, statuses: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "rank", "pair_id", "pep_id", "uniprot_id", "status", "returncode",
        "runtime_sec", "haddock_score", "vdw", "elec", "desolv", "bsa", "best_model",
        "pair_dir", "stdout_log", "stderr_log", "started_at", "finished_at",
    ]
    rows = sorted(statuses.values(), key=lambda row: (row["dataset"], int(row["rank"]), row["pair_id"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_one(row: dict[str, str], haddock_bin: str) -> dict[str, str]:
    pair_dir = Path(row["pair_dir"])
    stdout_log = pair_dir / "haddock_run.stdout.log"
    stderr_log = pair_dir / "haddock_run.stderr.log"
    score_file = final_score_file(pair_dir)
    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    with stdout_log.open("w") as out, stderr_log.open("w") as err:
        proc = subprocess.run(
            [haddock_bin, "haddock_peptide_score.cfg"],
            cwd=pair_dir,
            stdout=out,
            stderr=err,
            text=True,
        )

    finished = time.time()
    scores = parse_best_score(score_file)
    if proc.returncode == 0 and score_file.exists():
        status = "DONE"
    elif proc.returncode == 0:
        status = "NO_SCORE"
    else:
        status = "FAILED"

    return {
        "dataset": row["dataset"],
        "rank": row["rank"],
        "pair_id": row["pair_id"],
        "pep_id": row["pep_id"],
        "uniprot_id": row["uniprot_id"],
        "status": status,
        "returncode": str(proc.returncode),
        "runtime_sec": f"{finished - started:.1f}",
        "haddock_score": scores.get("haddock_score", ""),
        "vdw": scores.get("vdw", ""),
        "elec": scores.get("elec", ""),
        "desolv": scores.get("desolv", ""),
        "bsa": scores.get("bsa", ""),
        "best_model": scores.get("best_model", ""),
        "pair_dir": str(pair_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def short_job(row: dict[str, str]) -> str:
    return f"{row['dataset']}#{int(row['rank']):03d} {row['pep_id']}->{row['uniprot_id']}"


def fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "NA"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_process(pair_dir: Path) -> str:
    candidates = [pair_dir / "run_peptide_score" / "log", pair_dir / "haddock_run.stdout.log"]
    current = "starting"
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")[-20000:]
        except OSError:
            continue
        for line in text.splitlines():
            if "Running [" in line and "] module" in line:
                current = line.split("Running [", 1)[1].split("]", 1)[0]
            elif "Running haddock3-analyse" in line:
                current = "analyse"
            elif "Running haddock3-traceback" in line:
                current = "traceback"
            elif "Cleaning output" in line:
                current = "cleaning"
            elif "Finished at" in line:
                current = "finished"
    return current


def render_dashboard(
    args: argparse.Namespace,
    total_manifest: int,
    already_done: int,
    total_planned: int,
    completed: int,
    failed: int,
    pending: list[dict[str, str]],
    running: dict[object, dict[str, object]],
    started_all: float,
    last_finished: list[dict[str, str]],
) -> object:
    elapsed = time.time() - started_all
    done_total = already_done + completed
    running_count = len(running)
    pending_count = len(pending)
    avg = elapsed / completed if completed else None
    eta = avg * (total_planned - completed - running_count) if avg is not None else None

    config = Table.grid(padding=(0, 2))
    config.add_column(justify="left")
    config.add_column(justify="left")
    config.add_row("max_parallel", str(args.max_parallel))
    config.add_row("datasets", args.datasets)
    config.add_row("manifest rows", str(total_manifest))
    config.add_row("already done", str(already_done))
    config.add_row("status", str(args.status_csv))

    overall = Table.grid(padding=(0, 2))
    overall.add_column(justify="left")
    overall.add_column(justify="left")
    overall.add_row("done", f"{done_total} / {total_manifest}")
    overall.add_row("this run", f"{completed} / {total_planned}")
    overall.add_row("running", str(running_count))
    overall.add_row("failed", str(failed))
    overall.add_row("pending", str(pending_count))
    overall.add_row("elapsed", fmt_duration(elapsed))
    overall.add_row("avg/job", fmt_duration(avg))
    overall.add_row("eta", fmt_duration(eta))

    active = Table(box=box.ASCII, expand=True)
    active.add_column("Sample")
    active.add_column("Pair")
    active.add_column("Peptide")
    active.add_column("Protein")
    active.add_column("Process")
    active.add_column("Runtime")
    active.add_column("Cores")
    if running:
        values = sorted(running.values(), key=lambda item: (item["row"]["dataset"], int(item["row"]["rank"])))  # type: ignore[index]
        for meta in values:
            row = meta["row"]  # type: ignore[assignment]
            started = float(meta["started"])  # type: ignore[arg-type]
            pair_dir = Path(row["pair_dir"])  # type: ignore[index]
            active.add_row(
                str(row["dataset"]).upper(),  # type: ignore[index]
                f"{int(row['rank']):03d}/100",  # type: ignore[index]
                str(row["pep_id"]),  # type: ignore[index]
                str(row["uniprot_id"]),  # type: ignore[index]
                detect_process(pair_dir),
                fmt_duration(time.time() - started),
                "12",
            )
    else:
        active.add_row("-", "-", "-", "-", "idle", "00:00:00", "-")

    last = Table.grid()
    if last_finished:
        for row in last_finished[-3:]:
            score = row.get("haddock_score") or "NA"
            runtime = fmt_duration(float(row.get("runtime_sec") or 0))
            last.add_row(
                f"{row['dataset'].upper()} {int(row['rank']):03d}/100 "
                f"{row['pep_id']} -> {row['uniprot_id']} | score={score} | runtime={runtime}"
            )
    else:
        last.add_row("none yet")

    return Group(
        Text("PROGRAM DOCKING HADDOCK3", style="bold"),
        "",
        Panel(config, title="Config", box=box.ASCII),
        Panel(overall, title="Overall", box=box.ASCII),
        Panel(active, title="Active Jobs", box=box.ASCII),
        Panel(last, title="Last Finished", box=box.ASCII),
    )


def print_dry_run(args: argparse.Namespace, rows: list[dict[str, str]], planned: list[dict[str, str]], already_done: int) -> None:
    print("[DRY RUN] HADDOCK3 parallel runner")
    print(f"Manifest          : {args.manifest}")
    print(f"Status CSV        : {args.status_csv}")
    print(f"Rows in manifest  : {len(rows)}")
    print(f"Already done      : {already_done}")
    print(f"Planned jobs      : {len(planned)}")
    print(f"Max parallel      : {args.max_parallel}")
    print(f"HADDOCK bin       : {args.haddock_bin}")
    if planned:
        print("First jobs:")
        for row in planned[:5]:
            print(f"  {short_job(row)}")
    print("No jobs started.")


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    args = parse_args()
    rows = read_manifest(args.manifest)
    planned = filter_rows(rows, args)
    already_done = sum(1 for row in rows if is_done(row))
    use_live = RICH_AVAILABLE and not args.no_live

    if args.dry_run:
        print_dry_run(args, rows, planned, already_done)
        return 0
    if not planned:
        print("[DONE] Nothing to run.")
        return 0

    statuses = read_existing_status(args.status_csv)
    running: dict[object, dict[str, object]] = {}
    pending = list(planned)
    total = len(planned)
    completed = 0
    failed = 0
    started_all = time.time()
    last_finished: list[dict[str, str]] = []

    def maybe_render(live: Live | None) -> None:
        if live is not None:
            live.update(render_dashboard(args, len(rows), already_done, total, completed, failed, pending, running, started_all, last_finished))

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        live = Live(
            render_dashboard(args, len(rows), already_done, total, completed, failed, pending, running, started_all, last_finished),
            refresh_per_second=1,
            transient=False,
        ) if use_live else None
        if live is not None:
            live.start()
        try:
            while (pending or running) and not (STOP_REQUESTED and not running):
                while pending and len(running) < args.max_parallel and not STOP_REQUESTED:
                    row = pending.pop(0)
                    future = executor.submit(run_one, row, args.haddock_bin)
                    running[future] = {"row": row, "started": time.time()}
                    if not use_live:
                        print(f"[START] {short_job(row)} running={len(running)} pending={len(pending)}", flush=True)
                    maybe_render(live)

                if not running:
                    break

                done, _ = wait(running.keys(), timeout=args.poll_sec, return_when=FIRST_COMPLETED)
                if not done:
                    if not use_live:
                        elapsed_min = (time.time() - started_all) / 60
                        print(f"[LIVE] elapsed={elapsed_min:.1f}m running={len(running)} pending={len(pending)} done={completed}/{total}", flush=True)
                    maybe_render(live)
                    continue

                for future in done:
                    meta = running.pop(future)
                    row = meta["row"]  # type: ignore[assignment]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed += 1
                        result = {
                            "dataset": row["dataset"],
                            "rank": row["rank"],
                            "pair_id": row["pair_id"],
                            "pep_id": row["pep_id"],
                            "uniprot_id": row["uniprot_id"],
                            "status": "RUNNER_ERROR",
                            "returncode": "",
                            "runtime_sec": "",
                            "pair_dir": row["pair_dir"],
                            "stderr_log": repr(exc),
                        }
                    else:
                        completed += 1
                        if result["status"] != "DONE":
                            failed += 1

                    statuses[result["pair_id"]] = result
                    if result["status"] == "DONE":
                        last_finished.append(result)
                    write_status(args.status_csv, statuses)
                    if not use_live:
                        score = result.get("haddock_score") or "NA"
                        runtime = result.get("runtime_sec") or "NA"
                        print(
                            f"[{result['status']}] {short_job(row)} score={score} sec={runtime} "
                            f"done={completed}/{total} failed={failed}",
                            flush=True,
                        )
                    maybe_render(live)
        finally:
            if live is not None:
                live.stop()

    remaining = len(pending)
    print()
    print("[SUMMARY]")
    print(f"Completed this run : {completed}")
    print(f"Failed this run    : {failed}")
    print(f"Not started        : {remaining}")
    print(f"Status CSV         : {args.status_csv}")
    if STOP_REQUESTED:
        print("Stopped gracefully; rerun the same command to resume.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
