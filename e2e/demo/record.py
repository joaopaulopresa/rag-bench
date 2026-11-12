"""Record the demo video, the README animation and the screenshots.

Nothing is recorded by hand. The script runs the real commands in a pseudo-terminal, replays
their real output in a browser page, opens the real HTML report and films both with Playwright:

    uv run --group demo python e2e/demo/record.py                 # full demo of configs/demo.yaml
    uv run --group demo python e2e/demo/record.py --config configs/offline.yaml --out /tmp/demo

It needs the database up, a finished run of the configuration (so that the recorded run is
served from the cache and costs nothing), ``ffmpeg`` on the path and network access for the
terminal widget (xterm.js from a CDN).
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PLAYER = Path(__file__).with_name("player.html")
SIZE = {"width": 1280, "height": 720}
COLS, ROWS = 118, 28

CAPTION_STYLE = (
    "position:fixed;left:0;right:0;bottom:0;padding:14px 24px;text-align:center;z-index:99;"
    "font:600 21px system-ui,sans-serif;color:#f0efec;background:rgba(16,17,20,.92)"
)


def capture(command: list[str]) -> list[tuple[float, str]]:
    """Run ``command`` in a pseudo-terminal and return its output as timed chunks."""
    env = {**os.environ, "COLUMNS": str(COLS), "LINES": str(ROWS), "TERM": "xterm-256color"}
    pid, descriptor = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execvpe(command[0], command, env)
    started = time.perf_counter()
    events: list[tuple[float, str]] = []
    while True:
        ready, _, _ = select.select([descriptor], [], [], 0.5)
        if not ready:
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
            continue
        try:
            data = os.read(descriptor, 65536)
        except OSError:
            break
        if not data:
            break
        events.append((time.perf_counter() - started, data.decode("utf-8", "replace")))
    return events


def step(caption: str, command: str, *, speed: float = 1.0, hold: int = 2200) -> dict[str, object]:
    events = capture(command.split())
    text = "".join(data for _, data in events)
    if "error:" in text:
        sys.exit(f"`{command}` failed:\n{text}")
    return {
        "caption": caption,
        "command": command.removeprefix("uv run "),
        "events": events,
        "speed": speed,
        "maxGap": 900,
        "hold": hold,
    }


def set_caption(page: Page, text: str) -> None:
    page.evaluate(
        """([text, style]) => {
            let bar = document.getElementById('demo-caption');
            if (!bar) {
                bar = document.createElement('div');
                bar.id = 'demo-caption';
                bar.style.cssText = style;
                document.body.appendChild(bar);
            }
            bar.textContent = text;
        }""",
        [text, CAPTION_STYLE],
    )


def scroll_to(page: Page, heading: str, caption: str, pause: float) -> None:
    """Scroll smoothly until the section with this heading is at the top, then wait."""
    set_caption(page, caption)
    page.evaluate(
        """(heading) => {
            const target = [...document.querySelectorAll('h1, h2, .pick .label')]
                .find((node) => node.textContent.includes(heading));
            const top = target.getBoundingClientRect().top + window.scrollY - 24;
            const start = window.scrollY, distance = top - start, duration = 1400;
            const begin = performance.now();
            return new Promise((resolve) => {
                const frame = (now) => {
                    const t = Math.min(1, (now - begin) / duration);
                    const eased = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
                    window.scrollTo(0, start + distance * eased);
                    t < 1 ? requestAnimationFrame(frame) : resolve();
                };
                requestAnimationFrame(frame);
            });
        }""",
        heading,
    )
    page.wait_for_timeout(int(pause * 1000))


def film(
    cast: dict[str, object], report: Path, video_dir: Path, tour: list[tuple[str, str, float]]
) -> Path:
    """Film the terminal replay followed by a tour of the report; return the raw video."""
    video_dir.mkdir(parents=True, exist_ok=True)
    player = video_dir / "player.html"
    player.write_text(PLAYER.read_text().replace("/*CAST*/[]", json.dumps(cast)), encoding="utf-8")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport=SIZE, record_video_dir=str(video_dir), record_video_size=SIZE
        )
        page = context.new_page()
        page.goto(player.as_uri())
        page.wait_for_function("window.demoDone === true", timeout=180_000)
        page.goto(report.as_uri())
        page.add_style_tag(content="main { padding-bottom: 140px !important; }")
        for heading, caption, pause in tour:
            scroll_to(page, heading, caption, pause)
        context.close()
        browser.close()
        video = page.video
        assert video is not None
        return Path(video.path())


def screenshots(report: Path, out: Path) -> None:
    """Save still images of the main report sections."""
    sections = {
        "report-recommendation": "RAG benchmark report",
        "report-answers": "Stage 3",
        "report-reranking": "Stage 2",
        "report-retrieval": "Stage 1",
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=2)
        page.goto(report.as_uri())
        for name, heading in sections.items():
            found = page.evaluate(
                """(heading) => {
                    const target = [...document.querySelectorAll('h1, h2')]
                        .find((node) => node.textContent.includes(heading));
                    if (!target) return false;
                    window.scrollTo(0, target.getBoundingClientRect().top + window.scrollY - 24);
                    return true;
                }""",
                heading,
            )
            if found:
                page.screenshot(path=str(out / f"{name}.png"))
        browser.close()


def ffmpeg(*arguments: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--out", default=str(ROOT / "docs" / "media"))
    arguments = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required")
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    described = subprocess.run(
        ["uv", "run", "ragbench", "config", arguments.config],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"run id: (\S+)", described)
    assert match is not None
    run_id = match.group(1)
    report = ROOT / "reports" / run_id / "index.html"

    run = step(
        "One YAML file describes the experiment. A repeated run is served from the cache.",
        f"uv run ragbench run {arguments.config}",
        hold=3200,
    )
    make_report = step(
        "One command turns the stored results into leaderboards and a report.",
        f"uv run ragbench report {run_id}",
        hold=3000,
    )
    full_tour = [
        ("RAG benchmark report", "The report opens with one recommended configuration.", 6.0),
        ("Stage 3", "Quality against cost: every generator on the best retrieval setups.", 8.0),
        ("Judge validation", "The judge's scores are checked against human labels.", 5.0),
        ("Stage 2", "Rerankers: gain over the same questions, with a paired interval.", 6.0),
        ("Stage 1", "Chunking x embedding x search mode, on all questions.", 8.0),
        ("By source", "The best setups, broken down by document collection.", 5.0),
        ("RAG benchmark report", "Same config, same numbers: every table carries its run id.", 4.0),
    ]
    cast = {"cols": COLS, "rows": ROWS, "steps": [run, make_report]}
    raw = film(cast, report, work / "full", full_tour)
    ffmpeg(
        "-i",
        str(raw),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(out / "demo.mp4"),
    )

    # The README animation: the same real output, shorter, ending on the recommendation.
    short = {"cols": COLS, "rows": ROWS, "steps": [{**run, "hold": 1600, "maxGap": 500}]}
    raw_short = film(
        short,
        report,
        work / "short",
        [("RAG benchmark report", "", 3.0), ("Stage 3", "", 3.5)],
    )
    palette = work / "palette.png"
    filters = "fps=12,scale=960:-1:flags=lanczos"
    ffmpeg("-i", str(raw_short), "-vf", f"{filters},palettegen=max_colors=128", str(palette))
    ffmpeg(
        "-i",
        str(raw_short),
        "-i",
        str(palette),
        "-lavfi",
        f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
        str(out / "hero.gif"),
    )

    screenshots(report, out)
    shutil.rmtree(work)
    for file in sorted(out.iterdir()):
        print(f"{file}  {file.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
