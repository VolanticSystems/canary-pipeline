"""
Local orchestrator. Provisions a Vast.ai GPU instance, uploads the script and
audio, runs transcription remotely, pulls outputs back, destroys the instance.

Usage:
    python vast_provision.py <prepped.wav>
        [--gpu RTX_4090] [--image nvcr.io/nvidia/nemo:25.04] [--keep]
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

DEFAULT_IMAGE = "ghcr.io/bob7123/canary-asr:v1"
DEFAULT_DISK_GB = 50
DEFAULT_GPU = "RTX_4090"

# Floors for picking a host that's not bottom-of-the-barrel. These are the levers
# that prevent the "cheapest offer is cheapest for a reason" failure mode.
MIN_RELIABILITY = 0.985      # Vast 'reliability2', 0..1. Bumped from 0.97 after two host deaths
                             # during pulls (Elastrin's first host went offline mid-load 2026-07-09).
MIN_INET_DOWN_MBPS = 800     # Host's download bandwidth — drives image pull speed.
                             # Bumped from 500 for the same reason. Costs pennies more per hour.

REMOTE_FILES = [
    "canary_transcribe.py",
    "config.yaml",
]

# Linger before destroy on success, so a human can SSH in and inspect output
# files / installed package versions / GPU state before the instance is gone.
# Several times the polling interval per user request 2026-06-25.
LINGER_BEFORE_DESTROY_SECONDS = 5 * 60


def log(msg):
    print(f"[provision {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vastai_path():
    found = shutil.which("vastai")
    if found:
        return found
    fallback = r"C:\Users\Bob\AppData\Roaming\Python\Python312\Scripts\vastai.exe"
    if Path(fallback).is_file():
        return fallback
    raise RuntimeError("vastai CLI not found on PATH or at the fallback location")


def vastai(*args, capture=True):
    cmd = [vastai_path()] + list(args)
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"vastai failed: {' '.join(args)} (exit {r.returncode})")
    return r.stdout


def find_cheapest_offer(gpu, disk_gb):
    """Pick the cheapest RTX 4090 offer that's NOT a bottom-of-the-barrel host.
    'reliability2' and 'inet_down' are not valid as server-side filters, so we
    fetch a wider set and filter client-side.

    'datacenter=true' IS a valid server-side filter (2026-07-15) — restricts
    to vetted datacenter providers rather than random/residential hosts,
    which is a different axis than reliability2 (a historical uptime score
    that a residential host can still score well on). Vast's own docs
    recommend this for "Secure Cloud"-equivalent host selection. Falls back
    to non-datacenter hosts if the datacenter-only search comes back empty
    (some GPU/region combos may have no datacenter hosts available)."""
    base_flt = f"gpu_name={gpu} num_gpus=1 disk_space>={disk_gb} verified=true rentable=true"
    flt = base_flt + " datacenter=true"
    out = vastai("search", "offers", flt, "--raw")
    offers = json.loads(out)
    if not offers:
        log("  WARN: no datacenter=true offers found, falling back to all verified hosts")
        out = vastai("search", "offers", base_flt, "--raw")
        offers = json.loads(out)
    if not offers:
        raise RuntimeError(f"No offers matched server-side filter: {base_flt}")

    def keep(o):
        rel = o.get("reliability2") or 0
        inet = o.get("inet_down") or 0
        return rel >= MIN_RELIABILITY and inet >= MIN_INET_DOWN_MBPS

    quality = [o for o in offers if keep(o)]
    log(f"{len(offers)} raw offers, {len(quality)} pass quality floor "
        f"(reliability>={MIN_RELIABILITY}, inet_down>={MIN_INET_DOWN_MBPS} Mbps)")
    if not quality:
        log(f"  WARN: no quality-floor offers, falling back to verified-only set")
        quality = offers

    quality.sort(key=lambda o: o.get("dph_total", 999))
    o = quality[0]
    log(f"Picked offer: id={o['id']} gpu={o.get('gpu_name')} "
        f"dph=${o.get('dph_total', 0):.4f} disk={o.get('disk_space', 0)}GB "
        f"reliability={o.get('reliability2', 0):.3f} inet_down={o.get('inet_down', 0)}Mbps")
    return o


def create_instance(offer_id, image, disk_gb, hf_token):
    env_str = f"-e HF_TOKEN={hf_token} -e HUGGING_FACE_HUB_TOKEN={hf_token}"
    out = vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--env", env_str,
        "--ssh",
        "--raw",
    )
    info = json.loads(out)
    instance_id = info.get("new_contract") or info.get("instance_id") or info.get("id")
    if not instance_id:
        raise RuntimeError(f"could not parse instance id from: {info}")
    log(f"Created instance: {instance_id}")
    return instance_id


class StuckHostError(RuntimeError):
    """Raised when a host is confirmed bad (dead, or retry-looping on a pull)
    rather than just slow. Distinct from a plain timeout so callers can
    auto-reroll instead of giving up. Added 2026-07-15 after two separate
    incidents this week (Elastrin: host went offline mid-pull; Conference:
    host retry-looped on one Docker layer for 6+ minutes) where a human had
    to notice and manually destroy+retry."""
    pass


# How many consecutive polls of an UNCHANGED, "Retrying"-flavored status_msg
# before we call it stuck rather than slow. At the 10s poll interval below,
# 4 polls = 40s of zero progress on the same layer.
STUCK_RETRY_POLL_THRESHOLD = 4


def wait_for_ssh(instance_id, timeout=1800):
    """Poll until SSH is ready, OR raise StuckHostError early if the host is
    confirmed bad. Confirmed bad = either:
      (a) actual_status flips to 'offline' while we expected 'running', or
      (b) status_msg is unchanged and looks like a retry loop for
          STUCK_RETRY_POLL_THRESHOLD consecutive polls.
    A plain slow-but-progressing pull (different status_msg each poll) is
    never flagged stuck — only genuine no-progress patterns are."""
    log(f"Waiting for instance {instance_id} to be ready (timeout {timeout}s)...")
    deadline = time.time() + timeout
    last_status = None
    last_msg = None
    stuck_count = 0
    while time.time() < deadline:
        try:
            out = vastai("show", "instance", str(instance_id), "--raw")
            info = json.loads(out)
            status = info.get("actual_status", "unknown")
            if status != last_status:
                log(f"  status: {status}")
                last_status = status

            if status == "offline":
                raise StuckHostError(
                    f"Instance {instance_id} went offline while loading — host died mid-pull.")

            if status == "running":
                ssh_host = info.get("ssh_host")
                ssh_port = info.get("ssh_port")
                if ssh_host and ssh_port:
                    return ssh_host, int(ssh_port)

            msg = (info.get("status_msg") or "").strip().split("\n")[-1]
            looks_retrying = "retrying" in msg.lower()
            if msg and msg == last_msg and looks_retrying:
                stuck_count += 1
                log(f"  stuck watch: unchanged retry message x{stuck_count} "
                    f"({msg[:80]})")
                if stuck_count >= STUCK_RETRY_POLL_THRESHOLD:
                    raise StuckHostError(
                        f"Instance {instance_id} retry-looping on the same layer "
                        f"for {stuck_count} consecutive polls: {msg[:120]}")
            else:
                stuck_count = 0
            last_msg = msg
        except StuckHostError:
            raise
        except Exception as e:
            log(f"  poll error: {e}")
        time.sleep(10)
    raise RuntimeError(f"Instance {instance_id} did not become SSH-ready in {timeout}s")


def _ssh_base_opts(port):
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        # ServerAlive keepalives prevent NAT/firewall idle-kills like the one
        # that ended PeteDepo run #6 at ~50 min.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
        "-p", str(port),
    ]


def ssh_run(host, port, command, check=True, capture=False):
    cmd = ["ssh"] + _ssh_base_opts(port) + [f"root@{host}", command]
    if not capture:
        preview = command[:80] + ("..." if len(command) > 80 else "")
        log(f"$ ssh -p {port} root@{host} '{preview}'")
    return subprocess.run(cmd, check=check, capture_output=capture, text=capture)


def ssh_run_silent(host, port, command):
    """Run a remote command, capture stdout/stderr/rc, never raise."""
    return ssh_run(host, port, command, check=False, capture=True)


def build_launch_command(audio_name, base_stem, hf_token):
    """Build the remote shell command that launches the detached python.

    Filenames with spaces (e.g. "Paul Deposition.prepped.wav") must be quoted
    or the shell splits them into multiple args. This function exists as a
    standalone for unit-testing — the launch shell-quoting bug bit us twice
    before this was extracted."""
    audio_q = shlex.quote(audio_name)
    flag_q = shlex.quote(f"{base_stem}.done.flag")
    return (
        f"cd /root/work && "
        f"rm -f run.log {flag_q} && "
        f"(HF_TOKEN={hf_token} nohup python -u canary_transcribe.py {audio_q} "
        f"  < /dev/null > run.log 2>&1 &) && "
        f"sleep 2 && "
        f"echo 'launched pid:' && pgrep -f canary_transcribe.py"
    )


def build_poll_command(base_stem):
    """Build the remote shell command for one poll iteration.

    Same shell-quoting requirement as build_launch_command — base_stem may
    contain spaces."""
    done_flag_remote_q = shlex.quote(f"/root/work/{base_stem}.done.flag")
    base_stem_q = shlex.quote(base_stem)
    return (
        f"if [ -f {done_flag_remote_q} ]; then echo DONE; "
        f"elif pgrep -f canary_transcribe.py >/dev/null; then echo RUNNING; "
        f"else echo DEAD; fi; "
        f"echo '---progress---'; "
        f"for n in canary parakeet; do "
        f"  f=/root/work/{base_stem_q}.$n.progress.log; "
        f"  if [ -f \"$f\" ]; then echo \"== $n ==\"; tail -3 \"$f\"; fi; "
        f"done"
    )


def wait_ssh_responsive(host, port, timeout=600):
    log("Waiting for SSH to actually accept commands...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["ssh"] + _ssh_base_opts(port) + [
                    "-o", "ConnectTimeout=10",
                    f"root@{host}", "echo ready",
                ],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0 and "ready" in r.stdout:
                log("  SSH responsive")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("SSH never became responsive")


def scp_up(host, port, local_path, remote_path):
    scp = "scp"
    cmd = [scp] + [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-P", str(port),
        str(local_path),
        f"root@{host}:{remote_path}",
    ]
    log(f"$ scp -P {port} {Path(local_path).name} -> root@{host}:{remote_path}")
    subprocess.run(cmd, check=True)


def scp_down(host, port, remote_path, local_path):
    cmd = ["scp"] + [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-P", str(port),
        f"root@{host}:{remote_path}",
        str(local_path),
    ]
    log(f"$ scp -P {port} root@{host}:{remote_path} -> {local_path}")
    return subprocess.run(cmd, check=False).returncode == 0


def destroy_instance(instance_id):
    log(f"Destroying instance {instance_id}")
    try:
        vastai("destroy", "instance", str(instance_id), "-y")
    except Exception as e:
        log(f"  destroy error (you may need to remove it manually): {e}")


MAX_HOST_RETRIES = 2  # total attempts = 1 + this. "Vast sends us a turd about
                      # half the time" per user, 2026-07-15 — don't make a
                      # human notice and intervene, just get a new host.


def provision_with_retry(gpu, disk, image, hf_token, max_retries=MAX_HOST_RETRIES):
    """Create an instance and wait for it to be truly ready (SSH up AND
    responsive). If the host turns out to be a dud — confirmed via
    StuckHostError, or fails to become SSH-responsive in time — destroy it
    and try again on a fresh offer, up to max_retries times.

    Returns (instance_id, host, port) for a host confirmed good.
    Raises RuntimeError if all attempts are exhausted.
    """
    attempt = 0
    last_error = None
    while attempt <= max_retries:
        attempt += 1
        offer = find_cheapest_offer(gpu, disk)
        instance_id = create_instance(offer["id"], image, disk, hf_token)
        try:
            host, port = wait_for_ssh(instance_id)
            log(f"SSH endpoint: root@{host}:{port}")
            wait_ssh_responsive(host, port)
            return instance_id, host, port
        except (StuckHostError, RuntimeError) as e:
            last_error = e
            log(f"Host attempt {attempt}/{max_retries + 1} failed: {e}")
            log(f"  Destroying bad instance {instance_id} and trying a fresh host "
                f"({'retries left: ' + str(max_retries + 1 - attempt) if attempt <= max_retries else 'out of retries'})")
            destroy_instance(instance_id)
    raise RuntimeError(
        f"All {max_retries + 1} host attempts failed. Last error: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to prepped 16k mono WAV")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    parser.add_argument("--keep", action="store_true",
                        help="Leave instance up after run (manual destroy required)")
    parser.add_argument("--resume", type=int, default=None,
                        help="Attach to an existing instance instead of creating a new one")
    parser.add_argument("--max-host-retries", type=int, default=MAX_HOST_RETRIES,
                        help="How many times to destroy-and-reroll on a bad host before giving up")
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.is_file():
        print(f"ERROR: not found: {audio}", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("VAST_API_KEY"):
        print("WARNING: VAST_API_KEY not in this shell's env (CLI uses ~/.config/vastai/vast_api_key)",
              file=sys.stderr)

    if args.resume:
        # Explicit --resume means the human picked this instance on purpose —
        # no auto-reroll. If it's bad, that's on the human to destroy and
        # re-fire without --resume.
        instance_id = args.resume
        log(f"Resuming existing instance: {instance_id}")
        host, port = wait_for_ssh(instance_id)
        log(f"SSH endpoint: root@{host}:{port}")
        wait_ssh_responsive(host, port)
    else:
        instance_id, host, port = provision_with_retry(
            args.gpu, args.disk, args.image, hf_token, args.max_host_retries)

    success = False
    try:
        ssh_run(host, port, "mkdir -p /root/work")

        for fn in REMOTE_FILES:
            scp_up(host, port, HERE / fn, f"/root/work/{fn}")
        scp_up(host, port, audio, f"/root/work/{audio.name}")

        # The slim image already has nemo + pyannote + webrtcvad + soundfile + PyYAML
        # baked in. requirements-vast.txt is uploaded for parity but not installed
        # here (pip install would be a no-op on the slim image).

        # The in-image fixes (torch trio realign + pyannote<4 pin) are baked
        # into ghcr.io/bob7123/canary-asr:v1 starting with the rebuild on
        # 2026-06-25. No in-place pip surgery here anymore.

        # Launch the python DETACHED so SSH drops don't kill it. The previous
        # run died at 50 min because the host's TCP idle-killed our ssh pipe,
        # taking the python with it. Now: nohup + subshell-background, SSH
        # closes immediately after launch, python keeps running on the box.
        log("Launching transcription detached")
        base_stem = audio.stem  # e.g. "PeteDepo.prepped" or "Paul Deposition.prepped"
        launch = build_launch_command(audio.name, base_stem, hf_token)
        r = ssh_run_silent(host, port, launch)
        log(r.stdout.strip() if r.stdout else "(no launch output)")
        if r.returncode != 0:
            log(f"  launch returned rc={r.returncode}, stderr={r.stderr[:300]}")
            raise RuntimeError("Failed to launch remote python")

        # Poll for the done flag. While polling, pull down per-model progress
        # logs so the user can watch the rate live. Survives SSH drops because
        # each poll opens a fresh short SSH session.
        log("Polling for completion (status every 30s)")
        poll_cmd = build_poll_command(base_stem)
        poll_interval = 30
        polls = 0
        last_progress_summary = ""
        while True:
            polls += 1
            check = ssh_run_silent(host, port, poll_cmd)
            if check.returncode != 0:
                log(f"  poll {polls}: ssh poll failed (rc={check.returncode}); retrying")
                time.sleep(poll_interval)
                continue
            text = check.stdout or ""
            status_line = text.splitlines()[0] if text else "UNKNOWN"
            log(f"  poll {polls}: {status_line}")
            # Print progress summary if changed
            prog = text.split("---progress---", 1)[1].strip() if "---progress---" in text else ""
            if prog and prog != last_progress_summary:
                for line in prog.splitlines():
                    log(f"    {line}")
                last_progress_summary = prog

            if status_line == "DONE":
                log("Transcription complete (done.flag present)")
                break
            if status_line == "DEAD":
                log("Python process is no longer running but no done.flag — DIED")
                # Pull the run.log to disk for the user to inspect
                local_runlog = audio.parent / f"{base_stem}.run.log"
                scp_down(host, port, "/root/work/run.log", local_runlog)
                log(f"  remote run.log saved to {local_runlog}")
                break

            time.sleep(poll_interval)

        # Whatever happened, pull every output that exists. scp returns
        # False on missing files; that's fine, we just collect what's there.
        log("Downloading outputs")
        base = audio.stem
        out_dir = audio.parent
        # File taxonomy (post-Elastrin fix, 2026-07-10 → download-list catch-up 2026-07-14):
        # SHARED: written once per run, cover diagnostics and cross-model checks.
        shared_suffixes = [".diarization.json", ".coverage.json",
                           ".warnings.json", ".semantic_inversions.json"]
        # PER-MODEL: written once per {canary, parakeet}.
        per_model_suffixes = [".transcript.txt", ".srt", ".segments.json",
                              ".hallucinations.json",
                              ".final.txt", ".final.jsonl", ".speakers.json"]
        # DEBUG: always-try, useful even on partial runs.
        debug_suffixes = [".canary.progress.log", ".parakeet.progress.log",
                          ".canary.checkpoint.json", ".parakeet.checkpoint.json"]

        models = ["canary", "parakeet"]
        download_files = [f"{base}{sfx}" for sfx in shared_suffixes]
        for m in models:
            for sfx in per_model_suffixes:
                download_files.append(f"{base}.{m}{sfx}")
        download_files += [f"{base}{sfx}" for sfx in debug_suffixes]

        critical_files = [
            f"{base}{sfx}" for sfx in shared_suffixes
        ] + [
            f"{base}.{m}{sfx}" for m in models for sfx in per_model_suffixes
        ]

        present = []
        for fn in download_files:
            ok = scp_down(host, port, f"/root/work/{fn}", out_dir / fn)
            if ok:
                present.append(fn)
        # Also grab run.log for the record
        scp_down(host, port, "/root/work/run.log", out_dir / f"{base}.run.log")

        missing_critical = [fn for fn in critical_files if fn not in present]
        if not missing_critical:
            log(f"All {len(critical_files)} critical outputs downloaded")
            success = True
        else:
            log(f"Missing {len(missing_critical)} critical outputs:")
            for fn in missing_critical:
                log(f"  - {fn}")
            log("NOT destroying instance — leaving for inspection")

    finally:
        if args.keep:
            log(f"Leaving instance {instance_id} up (--keep).")
            log(f"  Destroy manually with: vastai destroy instance {instance_id} -y")
        elif success:
            log(f"Lingering {LINGER_BEFORE_DESTROY_SECONDS} s before destroy for diagnostics")
            log(f"  SSH while you can: ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
            log(f"  Or destroy now manually: vastai destroy instance {instance_id} -y")
            time.sleep(LINGER_BEFORE_DESTROY_SECONDS)
            destroy_instance(instance_id)
        else:
            # Lesson learned: don't destroy a still-progressing or partially-broken
            # instance. The previous run killed a host mid-image-pull and we lost
            # 25 min of work. Leave it for the human.
            log(f"Run did not complete successfully. Leaving instance {instance_id} up.")
            log(f"  Inspect: vastai ssh-url {instance_id}")
            log(f"  Destroy when ready: vastai destroy instance {instance_id} -y")


if __name__ == "__main__":
    main()
