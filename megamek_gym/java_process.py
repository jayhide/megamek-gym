"""Manage the MegaMek Java subprocess."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# JVM options matching rlJvmOptions in megamek/build.gradle
_JVM_OPTIONS = [
    "-Xmx1536m",
    "--add-opens", "java.base/java.util=ALL-UNNAMED",
    "--add-opens", "java.base/java.util.concurrent=ALL-UNNAMED",
    "-Dlog4j2.configurationFile=mmconf/log4j2-rl.xml",
    "-XX:+ExitOnOutOfMemoryError",
    "-XX:+HeapDumpOnOutOfMemoryError",
    "-XX:HeapDumpPath=rl_heapdump.hprof",
    "-Xlog:gc*:file=rl_gc.log:time,level,tags",
]


class JavaProcess:
    """Wraps a MegaMek RLGameRunner subprocess."""

    _cached_classpath: str | None = None

    def __init__(
        self,
        megamek_dir: str,
        rl_unit: str,
        opponent_unit: str,
        board: str,
        port: int,
        timeout_minutes: int,
        max_rotating_round_saves: int = 100,
        paranoid_autosave: bool = False,
        rl_starting_pos: int = 2,
        opponent_starting_pos: int = 6,
        rl_deployment: bool = False,
        firing_strategy: str = "princess",
    ):
        self.megamek_dir = Path(megamek_dir).resolve()
        self.rl_unit = rl_unit
        self.opponent_unit = opponent_unit
        self.board = board
        self.port = port
        self.timeout_minutes = timeout_minutes
        self.max_rotating_round_saves = max_rotating_round_saves
        self.paranoid_autosave = paranoid_autosave
        self.rl_starting_pos = rl_starting_pos
        self.opponent_starting_pos = opponent_starting_pos
        self.rl_deployment = rl_deployment
        self.firing_strategy = firing_strategy
        self._process: subprocess.Popen | None = None
        self._stderr_file = None

    @classmethod
    def _resolve_classpath(cls, megamek_dir: Path) -> str:
        """Resolve the runtime classpath via Gradle (cached after first call)."""
        if cls._cached_classpath is not None:
            return cls._cached_classpath

        init_script = Path(__file__).parent.parent / "scripts" / "write_classpath.init.gradle"
        cp_file = megamek_dir / "megamek" / "rl_classpath.txt"

        logger.info("Resolving classpath via Gradle (one-time)...")
        result = subprocess.run(
            [
                "./gradlew",
                "--no-daemon",
                "--init-script", str(init_script),
                ":megamek:writeRLClasspath",
                "-q",
            ],
            cwd=megamek_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Gradle classpath resolution failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        classpath = cp_file.read_text().strip()
        if not classpath:
            raise RuntimeError(f"Classpath file is empty: {cp_file}")

        cls._cached_classpath = classpath
        logger.info("Classpath resolved (%d entries)", classpath.count(":") + 1)
        return classpath

    def start(self) -> None:
        classpath = self._resolve_classpath(self.megamek_dir)

        args = [
            self.rl_unit,
            self.opponent_unit,
            self.board,
            str(self.port),
            str(self.timeout_minutes),
            str(self.max_rotating_round_saves),
            str(self.paranoid_autosave).lower(),
            str(self.rl_starting_pos),
            str(self.opponent_starting_pos),
            str(self.rl_deployment).lower(),
            self.firing_strategy,
        ]
        cmd = [
            "java",
            *_JVM_OPTIONS,
            "-cp", classpath,
            "megamek.client.bot.rl.RLGameRunner",
            *args,
        ]
        log_path = self.megamek_dir / f"rl_java_{self.port}.log"
        logger.info("Starting JVM directly on port %d (log: %s)", self.port, log_path)
        self._stderr_file = open(log_path, "w")
        # Each JVM gets its own CWD to avoid shared filesystem contention
        # (savegames dir, config writes, etc.)
        base_cwd = self.megamek_dir / "megamek"
        jvm_cwd = base_cwd / f"run_{self.port}"
        jvm_cwd.mkdir(exist_ok=True)
        # Symlink required subdirectories so relative paths resolve
        for subdir in ("mmconf", "data", "docs"):
            link = jvm_cwd / subdir
            target = base_cwd / subdir
            if not link.exists() and target.exists():
                link.symlink_to(target)
        self._process = subprocess.Popen(
            cmd,
            cwd=jvm_cwd,
            stdout=self._stderr_file,  # JVM thread dumps (SIGQUIT) go to stdout
            stderr=subprocess.STDOUT,   # Merge stderr (Log4j2) into same file
            start_new_session=True,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            exit_code = self._process.poll()
            logger.info("[stop:%d] poll() = %s", self.port, exit_code)
            if exit_code is not None:
                logger.info("[stop:%d] already exited", self.port)
            else:
                try:
                    pgid = os.getpgid(self._process.pid)
                except OSError:
                    pgid = None
                logger.info("[stop:%d] pgid=%s, pid=%s", self.port, pgid, self._process.pid)
                # Send SIGQUIT first to get a JVM thread dump (written to stderr/log)
                try:
                    os.kill(self._process.pid, signal.SIGQUIT)
                    logger.info("[stop:%d] SIGQUIT sent (thread dump requested)", self.port)
                    # Give JVM a moment to write the thread dump
                    try:
                        self._process.wait(timeout=2)
                        logger.info("[stop:%d] exited after SIGQUIT", self.port)
                        return  # JVM exited on its own
                    except subprocess.TimeoutExpired:
                        pass  # Expected — SIGQUIT doesn't kill the JVM
                except (ProcessLookupError, OSError):
                    logger.info("[stop:%d] SIGQUIT: process already gone", self.port)
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                        logger.info("[stop:%d] SIGTERM sent to pgid %s", self.port, pgid)
                    except ProcessLookupError:
                        logger.info("[stop:%d] SIGTERM: process group already gone", self.port)
                try:
                    self._process.wait(timeout=5)
                    logger.info("[stop:%d] exited after SIGTERM", self.port)
                except subprocess.TimeoutExpired:
                    logger.info("[stop:%d] SIGTERM timeout, sending SIGKILL", self.port)
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        self._process.kill()
                    logger.info("[stop:%d] final wait() START", self.port)
                    self._process.wait()
                    logger.info("[stop:%d] final wait() DONE", self.port)
        except Exception as e:
            logger.warning("Error stopping Java process on port %d: %s", self.port, e)
        finally:
            self._process = None
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
            logger.info("[stop:%d] COMPLETE", self.port)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None
