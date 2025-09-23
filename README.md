# Diskmon

**Diskmon** is a Linux disk health monitoring tool designed to check the status of block devices, run SMART tests, track usage/temperature, and push results to [Uptime Kuma](https://github.com/louislam/uptime-kuma).

It balances robustness, maintainability, and elegance, while supporting both **ATA/SATA** and **NVMe** devices.

---

## Features

* **SMART monitoring** for ATA and NVMe drives
* Automatic **short/long test scheduling**
* Optional **badblocks scans**
* **Filesystem usage** monitoring with thresholds
* **Idle-aware checks** to avoid unnecessary spin-ups
* **Persistent state tracking** with `filelock`
* **Color-coded human-readable output**, JSON mode, or Uptime Kuma only
* **Logging** with rotation (`/var/log/diskmon.log` by default)

---

## Requirements

### Dependencies

* Python 3.7+
* External tools:

  * `smartctl` (from `smartmontools`)
  * `badblocks` (optional, for surface scans)

### Python Packages

Install with:

```bash
pip3 install requests filelock psutil
```

---

## Usage

Basic usage (requires root privileges):

```bash
sudo ./diskmon.py /dev/sdX --kuma-url http://kuma.local/push/XXXXX
```

### Common Options

* `--usage-threshold <percent>`: Filesystem usage alert threshold (default: 90%)
* `--temp-threshold <C>`: Temperature threshold (default: 50°C)
* `--short-test-days <n>`: Days between short SMART tests (default: 7)
* `--long-test-days <n>`: Days between long SMART tests (default: 30)
* `--badblocks-days <n>`: Days between badblocks scans (optional)
* `--output {human,json,kuma-only}`: Output format (default: `human`)
* `--verbose`: Enable detailed logging
* `--no-color`: Disable colored output

### Example

Run against `/dev/nvme0n1`, pushing to Kuma, with a higher temperature limit:

```bash
sudo ./diskmon.py /dev/nvme0n1 --kuma-url http://kuma.local/push/XXXXX --temp-threshold 60
```

---

## Configuration File

Diskmon supports a config file (default: `~/.config/diskmon.conf`), e.g.:

```ini
[diskmon]
usage_threshold = 85
temp_threshold = 55
short_test_interval = 5
long_test_interval = 20
log_file = /var/log/diskmon.log

[smart_thresholds]
pending = 1
realloc = 5
uncorrectable = 0
reported_uncorrect = 0
```

---

## Integration with Uptime Kuma

Diskmon pushes results directly to Kuma’s HTTP(s) push monitors, reporting:

* Status: `up` or `down`
* Message: latest health summary
* Ping: duration of the health check in ms

---

## Disclaimer ⚠️

Like most of the code on Github these days, this script was **completely vibe coded** (and so was the README as I'm sure you can tell).
While it works in practice, **human eyes have not reviewed it in depth**.

👉 Use **at your own risk**.


