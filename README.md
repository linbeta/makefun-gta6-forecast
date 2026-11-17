# makefun-gta6-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/linbeta/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/linbeta/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour forecast of GTA VI launch-day Twitch concurrent viewers (hours 2–24 of 2026-11-19 UTC) for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty. The program is standard-library Python 3, deterministic, and uses no network at runtime.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

Cases are UTF-8 JSON in `$MAKEFUN_INPUT_DIR` or `/input` (schema `makefun/twitch-ccu-case/v1`). Each forecast is written to `$MAKEFUN_OUTPUT_DIR/<id>.out` or `/output/<id>.out` as `{"hours":[{"hour":h,"q":[q05,q25,q50,q75,q95]}, ...]}` for hours 2–24.

Claude Code session transcripts used to build this entry are in `transcripts/`, copied unmodified from the harness logs.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
