#!/usr/bin/env python3
"""Show GPU processes with service names and VRAM usage."""

import re
import os
import subprocess
import xml.etree.ElementTree as ET


def get_all_gpu_processes():
    """Return list of (pid, type, vram_mib) for all GPU processes via nvidia-smi XML."""
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True)
    root = ET.fromstring(result.stdout)
    processes = []
    for proc in root.findall(".//process_info"):
        pid = int(proc.findtext("pid"))
        proc_type = proc.findtext("type", "")            # "C", "G", or "C+G"
        mem_text = proc.findtext("used_memory", "0 MiB") # "760 MiB"
        vram_mib = int(mem_text.replace("MiB", "").strip())
        processes.append((pid, proc_type, vram_mib))
    return processes


def get_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline") as f:
            return f.read().replace("\x00", " ").strip()
    except FileNotFoundError:
        return None


def get_cwd(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError):
        return None


def parse_llama_model_name(cmdline):
    """Extract a human-readable model name from a llama-server command line."""
    m = re.search(r"--model\s+(\S+)", cmdline)
    if not m:
        return None
    filename = os.path.basename(m.group(1))
    # Strip .gguf and quantization suffixes like -q4_k_m, _q4_k_m, -Q4_K_M
    name = re.sub(r"\.gguf$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"[-_][qQ]\d[_\-][kKmMlLsS][_\-]?[mMsSlL]?$", "", name)
    # Also strip generic quantization like -f16, -f32, -q8_0
    name = re.sub(r"[-_][fFqQ]\d+(?:_\d+)?$", "", name)
    return name


def identify_service(pid, cmdline, cwd):
    """Return a human-readable service name for a process."""
    if cmdline is None:
        return "(gone)"

    # llama-server: use model filename
    if "llama-server" in cmdline:
        model = parse_llama_model_name(cmdline)
        return model if model else "llama-server"

    # Named python environments (e.g. yolo-venv)
    m = re.match(r"(\S+venv\S*)/bin/python", cmdline)
    if m:
        venv_name = os.path.basename(m.group(1).rstrip("/"))
        # Strip "-venv" suffix for cleanliness
        return re.sub(r"-venv$", "", venv_name)

    # Fall back to working directory name
    if cwd:
        return os.path.basename(cwd)

    return cmdline.split()[0] if cmdline else "?"


def main():
    all_procs = get_all_gpu_processes()
    if not all_procs:
        print("No GPU processes found.")
        return

    compute_rows = []
    display_vram = 0

    for pid, proc_type, vram_mib in all_procs:
        if proc_type == "G":
            display_vram += vram_mib
            continue

        cmdline = get_cmdline(pid)
        cwd = get_cwd(pid)
        service = identify_service(pid, cmdline, cwd)
        compute_rows.append((service, pid, vram_mib))

    compute_rows.sort(key=lambda r: r[2], reverse=True)

    if display_vram:
        compute_rows.append(("Xorg + gnome-shell", "—", display_vram))

    total = sum(r[2] for r in compute_rows)

    svc_w  = max(len("Service"), max(len(str(r[0])) for r in compute_rows))
    pid_w  = max(len("PID"),     max(len(str(r[1])) for r in compute_rows))
    vram_w = max(len("VRAM"),    max(len(f"{r[2]:,} MiB") for r in compute_rows), len(f"{total:,} MiB"))

    def hr(left, mid, right):
        return left + "─" * (svc_w + 2) + mid + "─" * (pid_w + 2) + mid + "─" * (vram_w + 2) + right

    def row(svc, pid, vram_str):
        return f"│ {svc:<{svc_w}} │ {pid:>{pid_w}} │ {vram_str:>{vram_w}} │"

    print(hr("┌", "┬", "┐"))
    print(row("Service", "PID", "VRAM"))
    print(hr("├", "┼", "┤"))
    for i, (svc, pid, vram_mib) in enumerate(compute_rows):
        print(row(svc, str(pid), f"{vram_mib:,} MiB"))
        if i < len(compute_rows) - 1:
            print(hr("├", "┼", "┤"))
    print(hr("├", "┼", "┤"))
    print(row("Total", "", f"{total:,} MiB"))
    print(hr("└", "┴", "┘"))


if __name__ == "__main__":
    main()
