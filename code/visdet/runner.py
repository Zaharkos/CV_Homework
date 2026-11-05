import os
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent


def n_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU")])
    except Exception:
        return 0


def tail(path, n=15):
    try:
        lines = Path(path).read_text(errors="ignore").replace("\r", "\n").splitlines()
        return "\n".join(l for l in lines[-n:] if l.strip())
    except FileNotFoundError:
        return ""


def last_progress(path):
    try:
        lines = Path(path).read_text(errors="ignore").replace("\r", "\n").splitlines()
    except FileNotFoundError:
        return ""
    for l in reversed(lines):
        if l.startswith("[") or "Epoch" in l:
            return l.strip()[:160]
    return lines[-1].strip()[:160] if lines else ""


def reuse_finished(name, args, roots=("/kaggle/input",)):
    import glob
    import shutil
    out_root = Path(args[args.index("--out") + 1])
    for r in roots:
        for m in sorted(glob.glob(os.path.join(r, "**", name, "metrics.json"), recursive=True)):
            d = Path(m).parent
            if (d / "best.pt").exists() or (d / "model_final.pth").exists():
                shutil.copytree(d, out_root / name, dirs_exist_ok=True)
                return d
    return None


def run_jobs(jobs, log_dir, stagger_s=60, report_every_s=600):
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    gpus = list(range(n_gpus())) or [None]
    codes = {}
    todo = []
    for name, args in jobs:
        src = reuse_finished(name, args)
        if src is not None:
            print(f"{name}: reused finished run from {src}", flush=True)
            codes[name] = 0
        else:
            todo.append((name, args))
    queue, running = todo, {}
    last_report, last_start = time.time(), 0.0
    env0 = {**os.environ, "PYTHONPATH": str(CODE_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1"}
    while queue or running:
        free = [g for g in gpus if g not in running.values()]
        if queue and free and time.time() - last_start >= (stagger_s if running else 0):
            name, args = queue.pop(0)
            g = free[0]
            env = dict(env0)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            log = open(log_dir / f"{name}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", *args], cwd=CODE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[p] = g
            p._name, p._log = name, log
            last_start = time.time()
            print(f"{time.strftime('%H:%M:%S')} started {name} on GPU {g}", flush=True)
            continue
        for p in list(running):
            if p.poll() is not None:
                p._log.close()
                codes[p._name] = p.returncode
                del running[p]
                status = "OK" if p.returncode == 0 else f"FAILED ({p.returncode})"
                print(f"{time.strftime('%H:%M:%S')} {p._name}: {status}", flush=True)
                if p.returncode != 0:
                    print(tail(log_dir / f"{p._name}.log", 25), flush=True)
        if time.time() - last_report > report_every_s and running:
            for p in running:
                print(f"   {p._name}: {last_progress(log_dir / f'{p._name}.log')}", flush=True)
            last_report = time.time()
        time.sleep(5)
    return codes


def tv_job(exp, data_root, out_root, budget_h=None):
    a = ["visdet.tv_train", "--exp", exp, "--data", str(data_root), "--out", str(out_root)]
    return (exp, a + (["--budget-h", str(budget_h)] if budget_h else []))


def ul_job(exp, data_yaml, out_root, budget_h=None, cfg=None):
    import json
    a = ["visdet.ul_train", "--exp", exp, "--data", str(data_yaml), "--out", str(out_root)]
    if budget_h:
        a += ["--budget-h", str(budget_h)]
    if cfg:
        a += ["--cfg-json", json.dumps(cfg)]
    return (exp, a)
