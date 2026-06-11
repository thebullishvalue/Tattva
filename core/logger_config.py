"""
Tattva v2.0.0 — Logger configuration: direct console output system.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Bypasses Python logging, writes colored output to stdout for clean terminal analysis pipeline.

Bypasses Python logging and writes directly to stdout for clean,
colored terminal output during the analysis pipeline.
Adapted from the Pragyam 7.0.5 design.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Any

# ── ANSI color support ──────────────────────────────────────────────────────

try:
    import colorama

    colorama.init()
except ImportError:
    if os.name == "nt":
        from ctypes import windll, byref, c_ulong

        STD_OUTPUT_HANDLE = -11
        h_console = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = c_ulong()
        windll.kernel32.GetConsoleMode(h_console, byref(mode))
        mode.value |= 0x0004
        windll.kernel32.SetConsoleMode(h_console, mode)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Run identifier ──────────────────────────────────────────────────────────

_SESSION_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
_SESSION_UUID = str(uuid.uuid4())[:8]
SESSION_RUN_IDENTIFIER = f"{_SESSION_RUN_ID}_{_SESSION_UUID}"


def generate_run_id() -> str:
    """Generate a unique Run ID for each analysis run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]
    return f"{run_id}_{run_uuid}"


def get_run_id() -> str:
    """Get the session-level run identifier."""
    return SESSION_RUN_IDENTIFIER


# ── ANSI color codes ────────────────────────────────────────────────────────


class Colors:
    """ANSI color codes compatible with Windows 10+."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    SUCCESS = "✓"
    WARNING = "⚠"
    ERROR = "✗"
    INFO = "ℹ"


# ── Console output ──────────────────────────────────────────────────────────


class ConsoleOutput:
    """Direct console output with styled sections and progress tracking."""

    def __init__(self) -> None:
        self._section_depth = 0
        self._phase_timers: dict[str, datetime] = {}

    def _write(self, message: str = "", end: str = "\n") -> None:
        """Write directly to stdout with UTF-8 fallback."""
        text = f"{message}{end}"
        try:
            sys.stdout.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe = text.encode("utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
            sys.stdout.write(safe)
        sys.stdout.flush()

    def _timestamp(self) -> str:
        """Current HH:MM:SS timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self, start: datetime) -> str:
        """Elapsed time since a reference point."""
        elapsed = datetime.now() - start
        total_secs = int(elapsed.total_seconds())
        if total_secs < 60:
            return f"{total_secs}s"
        return f"{total_secs // 60}m {total_secs % 60}s"

    def line(self, char: str = "─", length: int = 70) -> None:
        """Print a separator line."""
        self._write(f"{Colors.GRAY}{char * length}{Colors.RESET}")

    def header(self, title: str, version: str = "") -> None:
        """Print the run header."""
        self._write()
        self.line("═", 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title} {version}{Colors.RESET}")
        self._write(f"  {Colors.GRAY}Run ID: {SESSION_RUN_IDENTIFIER}{Colors.RESET}")
        self._write(
            f"  {Colors.GRAY}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}"
        )
        self.line("═", 70)
        self._write()

    def main_header(self, title: str, details: dict[str, Any]) -> None:
        """Print a main header with key-value details."""
        self._write()
        self.line("═", 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
        self.line("─", 70)
        for key, value in details.items():
            self._write(f"  {Colors.GRAY}{key}:{Colors.RESET} {value}")
        self.line("═", 70)
        self._write()

    def section(self, title: str, phase: str = "") -> None:
        """Print a section header."""
        self._write()
        if phase:
            self.line("═", 70)
            self._write(
                f"  {Colors.BOLD}{Colors.BLUE}{phase}: {title}{Colors.RESET}"
            )
            self.line("═", 70)
        else:
            self._write(f"{Colors.BOLD}{title}{Colors.RESET}")
            self._write(Colors.GRAY + "─" * len(title) + Colors.RESET)
        self._section_depth += 1

    def step(self, num: int, title: str) -> None:
        """Print a numbered step."""
        self._write(f"  {Colors.BOLD}Step {num}:{Colors.RESET} {title}")

    def item(self, label: str, value: Any, indent: int = 4) -> None:
        """Print a labeled key-value pair."""
        self._write(f"{' ' * indent}{Colors.GRAY}{label}:{Colors.RESET} {value}")

    def detail(self, message: str) -> None:
        """Print a detail line with an arrow prefix."""
        self._write(f"    {Colors.CYAN}→{Colors.RESET} {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        self._write(
            f"  {Colors.GREEN}{Colors.SUCCESS} SUCCESS:{Colors.RESET} {message}"
        )

    def warning(self, message: str) -> None:
        """Print a warning message."""
        self._write(
            f"  {Colors.YELLOW}{Colors.WARNING} WARNING:{Colors.RESET} {message}"
        )

    def error(self, message: str) -> None:
        """Print an error message."""
        self._write(f"  {Colors.RED}{Colors.ERROR} ERROR:{Colors.RESET} {message}")

    def failure(self, step: str, error: str) -> None:
        """Print a failure with context."""
        self._write(f"  {Colors.RED}{Colors.ERROR} FAILURE:{Colors.RESET} {step}")
        self._write(f"      {Colors.GRAY}Reason:{Colors.RESET} {error}")

    def issue(self, issue_type: str, location: str, description: str) -> None:
        """Flag an issue."""
        self._write(
            f"  {Colors.YELLOW}{Colors.WARNING} ISSUE [{issue_type}]{Colors.RESET} "
            f"at {location}"
        )
        self._write(f"      {Colors.GRAY}{description}{Colors.RESET}")

    def checkpoint(self, name: str, status: str = "OK") -> None:
        """Print a checkpoint."""
        symbol = (
            Colors.GREEN + Colors.SUCCESS
            if status == "OK"
            else Colors.RED + Colors.ERROR
        )
        self._write(
            f"  {symbol} Checkpoint:{Colors.RESET} {name} "
            f"{Colors.GRAY}[{status}]{Colors.RESET}"
        )

    def summary(self, title: str, data: dict[str, Any]) -> None:
        """Print a boxed summary."""
        self._write()
        self._write(f"  {Colors.GRAY}┌─ {title}{Colors.RESET}")
        for key, value in data.items():
            self._write(f"  {Colors.GRAY}│   {key}:{Colors.RESET} {value}")
        self._write(f"  {Colors.GRAY}└─{Colors.RESET}")

    def start_phase(self, phase: str, num: int = 0, total: int = 0) -> None:
        """Start a timed phase."""
        self._phase_timers[phase] = datetime.now()
        if num and total:
            self._write()
            self.line("═", 70)
            self._write(
                f"  {Colors.BOLD}{Colors.BLUE}Phase {num}/{total}: {phase}{Colors.RESET}"
            )
            self.line("═", 70)
        else:
            self._write()
            self.line("─", 60)
            self._write(f"  {Colors.BOLD}{Colors.BLUE}Phase: {phase}{Colors.RESET}")
            self.line("─", 60)

    def end_phase(self, phase: str) -> None:
        """End a timed phase."""
        if phase in self._phase_timers:
            elapsed = self._elapsed(self._phase_timers[phase])
            self._write(
                f"\n  {Colors.GREEN}{Colors.SUCCESS} "
                f"Phase Complete: {phase} {Colors.GRAY}[{elapsed}]{Colors.RESET}"
            )
            del self._phase_timers[phase]


# ── Global instance ─────────────────────────────────────────────────────────

console = ConsoleOutput()


def get_console() -> ConsoleOutput:
    """Get the global console instance."""
    return console
