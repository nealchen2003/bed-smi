import asyncio
from datetime import datetime
from typing import NamedTuple
from rich.live import Live
from rich.table import Table

ServerInfo = NamedTuple("ServerInfo", [("address", str), ("check_err", bool)])

servers = {}
server_states = {}


async def get_gpu_async(name, server_info):
    if name not in server_states:
        server_states[name] = {"gpus": [], "status": "[dim]Waiting...[/dim]"}

    server_states[name]["status"] = "[yellow]Refreshing...[/yellow]"
    server, check_err = server_info.address, server_info.check_err

    if check_err:
        query_fields = "utilization.gpu,memory.used,memory.total,reset_status.reset_required"
    else:
        query_fields = "utilization.gpu,memory.used,memory.total"

    cmd = f'ssh -o ConnectTimeout=5 {server} "nvidia-smi --query-gpu={query_fields} --format=csv,noheader,nounits"'

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            server_states[name]["gpus"] = []
            server_states[name]["status"] = "[red]Timeout[/red]"
            return

        if proc.returncode == 0:
            result = stdout.decode().strip().split("\n")
            server_states[name]["gpus"] = result
            server_states[name]["last_updated"] = datetime.now()
            server_states[name]["status"] = "OK"
        else:
            server_states[name]["gpus"] = []
            server_states[name]["status"] = "[red]SSH Error[/red]"

    except Exception as e:
        server_states[name]["gpus"] = []
        server_states[name]["status"] = f"[red]{str(e)}[/red]"


async def server_loop(name, info):
    if name not in server_states:
        server_states[name] = {"gpus": [], "status": "[dim]Waiting...[/dim]"}

    while True:
        await get_gpu_async(name, info)
        await asyncio.sleep(5)


def wrap_mib_to_gib(mib_str):
    mib = int(mib_str)
    gib = mib / 1024
    return f"{gib:>4.1f}"


def make_table():
    table = Table("Server", "GPU", "Util", "Memory", "Status")

    for name, server_info in servers.items():
        state = server_states.get(name, {"gpus": [], "status": "[dim]Waiting...[/dim]"})
        status = state.get("status", "[dim]Unknown[/dim]")
        gpus = state.get("gpus", [])

        if status == "OK":
            last_updated = state.get("last_updated")
            if last_updated:
                delta = datetime.now() - last_updated
                seconds = int(delta.total_seconds())
                status = f"[green]{seconds}s ago[/green]"

        if not gpus:
            table.add_row(name, "...", "...", "...", status)
            continue

        for idx, gpu in enumerate(gpus):
            if not gpu.strip():
                continue

            if server_info.check_err:
                parts = gpu.split(", ")
                assert len(parts) == 4
                util, mem_used, mem_total, reset_required = parts[:4]
                broken = reset_required == "Yes"
            else:
                parts = gpu.split(", ")
                assert len(parts) >= 3
                util, mem_used, mem_total = parts[:3]
                broken = False
            if broken:
                util = memory = "[red]ERR[/red]"
            else:
                util += "%"
                memory = wrap_mib_to_gib(mem_used) + " / " + wrap_mib_to_gib(mem_total) + " GiB"

            # Only show status on the first row for this server
            row_status = status if idx == 0 else ""
            table.add_row(name if idx == 0 else "", str(idx), util, memory, row_status)

    return table


async def main():
    # Start separate update loops for each server
    for name, info in servers.items():
        asyncio.create_task(server_loop(name, info))

    with Live(make_table(), refresh_per_second=4, screen=True) as live:
        while True:
            live.update(make_table())
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    # parse a file argument
    import argparse

    argparser = argparse.ArgumentParser(description="GPU Monitor")
    argparser.add_argument("servers", nargs="?", default="servers", help="File containing server info")
    args = argparser.parse_args()
    with open(args.servers) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name, server = parts[:2]
            check_err = False
            for flag in parts[2:]:
                if flag.lower() == "check-err":
                    check_err = True
                else:
                    print(f"Unknown flag '{flag}' in line: {line}")
            servers[name] = ServerInfo(address=server, check_err=check_err)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
