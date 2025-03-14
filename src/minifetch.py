# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "distro",
#     "psutil",
#     "rich",
#     "screeninfo",
# ]
# ///

from datetime import datetime

import os
import subprocess
import platform

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.text import Text

import screeninfo
import distro
import psutil


def username() -> str:
    return os.getlogin()


def hostname() -> str:
    return platform.node()


def os_() -> str:
    return f"{distro.name(pretty=True)} {platform.processor()}"


def kernel() -> str:
    return platform.release()


def uptime() -> str:
    boot_time = psutil.boot_time()
    seconds = int(datetime.now().timestamp() - boot_time)

    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{hours}h {minutes}m {seconds}s"


def packages() -> str:
    command = "dpkg -l | wc -l"
    packages = (
        subprocess
        .check_output(command, shell=True, text=True)
        .strip()
    )

    return f"{packages} (dpkg)"


def shell() -> str:
    shell = os.environ.get("SHELL")
    return os.path.basename(shell) if shell else "N/A"


def resolution() -> str:
    monitors = screeninfo.get_monitors()
    return ', '.join(map(lambda m: f"{m.width}x{m.height}", monitors))


def desktop_environment() -> str:
    de = os.environ.get("DESKTOP_SESSION")
    return de if de else "N/A"


def window_manager() -> str:
    command = "wmctrl -m | grep Name"
    _, wm = (
        subprocess
        .check_output(command, shell=True, text=True)
        .split(": ")
    )

    return wm.strip()


def terminal() -> str:
    term = os.environ.get("TERM")
    return term if term else "N/A"


def cpu() -> str:
    command = "lscpu | grep 'Model name'"
    _, cpu = (
        subprocess
        .check_output(command, shell=True, text=True)
        .split(": ")
    )

    return cpu.strip()


def gpus() -> list[str]:
    command = "lspci | grep VGA"
    out = (
        subprocess
        .check_output(command, shell=True, text=True)
        .splitlines()
    )

    return [gpu.strip() for _, gpu in map(lambda line: line.split(": "), out)]


def memory() -> str:
    mem = psutil.virtual_memory()
    to_mib = lambda x: x // (1024 * 1024)

    return f"{to_mib(mem.used)}MiB / {to_mib(mem.total)}MiB"


def main() -> None:
    console = Console()
    table = Table(show_header=False, box=None)

    label_style = "bold cyan"
    values_style = "white"

    table.add_row(Text(s := f"{username()}@{hostname()}", style=label_style))
    table.add_row(Text("-" * len(s), style=label_style))
    table.add_row(Text("OS: ", style=label_style), Text(os_(), style=values_style))
    table.add_row(Text("Host: ", style=label_style), Text(hostname(), style=values_style))
    table.add_row(Text("Kernel: ", style=label_style), Text(kernel(), style=values_style))
    table.add_row(Text("Uptime: ", style=label_style), Text(uptime(), style=values_style))
    table.add_row(Text("Packages: ", style=label_style), Text(packages(), style=values_style))
    table.add_row(Text("Shell: ", style=label_style), Text(shell(), style=values_style))
    table.add_row(Text("Resolution: ", style=label_style), Text(resolution(), style=values_style))
    table.add_row(Text("DE: ", style=label_style), Text(desktop_environment(), style=values_style))
    table.add_row(Text("WM: ", style=label_style), Text(window_manager(), style=values_style))
    table.add_row(Text("Terminal: ", style=label_style), Text(terminal(), style=values_style))
    table.add_row(Text("CPU: ", style=label_style), Text(cpu(), style=values_style))

    for gpu in gpus():
        table.add_row(Text("GPU: ", style=label_style), Text(gpu, style=values_style))

    table.add_row(Text("Memory: ", style=label_style), Text(memory(), style=values_style))

    with open("../sources/minifetch_logo.txt") as file:
        logo_text = file.read()
        logo_panel = Panel(Text(logo_text, style="bold yellow", justify="center"), expand=False)

    layout = Table(show_header=False, box=None)
    layout.add_row(logo_panel, table)

    console.print(Align.left(layout))


if __name__ == "__main__":
    main()