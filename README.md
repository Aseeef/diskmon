# Diskmon

**Diskmon** is a Linux disk health monitoring tool that checks block devices and btrfs filesystems, then pushes results to [Uptime Kuma](https://github.com/louislam/uptime-kuma).

It supports **ATA/SATA**, **NVMe**, and **btrfs** targets.

---

## Features

* **SMART monitoring** for ATA and NVMe drives
* **Btrfs scrub monitoring** with automatic scrub scheduling
* Automatic **short/long SMART test scheduling**
* Optional **badblocks scans**
* **Filesystem usage** monitoring with thresholds
* **Idle-aware checks** to avoid unnecessary spin-ups (block devices)
* **Persistent state tracking** with `filelock`
* **Color-coded human-readable output**, JSON mode, or Uptime Kuma only
* **Logging** with rotation (`/var/log/diskmon.log` by default)

---

## Requirements

### Dependencies

* Python 3.7+
* External tools:

  * `smartctl` (from `smartmontools`) — for block device monitoring
  * `btrfs` (from `btrfs-progs`) — for btrfs monitoring
  * `badblocks` (optional, for surface scans)

### Python Packages

Install with:

```bash
pip3 install requests filelock psutil
```

---

## Usage

Diskmon auto-detects the monitoring mode based on the target path:

* **Block device** (`/dev/...`) — monitors via SMART (ATA/SATA/NVMe)
* **Directory** (mount path) — monitors via btrfs scrub

```bash
# Block device
sudo ./diskmon.py /dev/sda --kuma-url http://kuma.local/push/XXXXX

# Btrfs mount path (must be a directory, not a /dev path)
sudo ./diskmon.py /media/backups --kuma-url http://kuma.local/push/XXXXX
```

### Common Options

| Option | Description | Mode |
|---|---|---|
| `--usage-threshold <percent>` | Filesystem usage alert threshold (default: 90%) | Both |
| `--output {human,json,kuma-only}` | Output format (default: `human`) | Both |
| `--verbose` | Enable detailed logging | Both |
| `--no-color` | Disable colored output | Both |
| `--config <path>` | Configuration file path | Both |
| `--temp-threshold <C>` | Temperature threshold (default: 50°C) | Block device |
| `--smart <thresholds>` | Error thresholds (see below) | Both |
| `--short-test-days <n>` | Days between short SMART tests (default: 7) | Block device |
| `--long-test-days <n>` | Days between long SMART tests (default: 30) | Block device |
| `--badblocks-days <n>` | Days between badblocks scans (optional) | Block device |
| `--idle-threshold <sec>` | Seconds to consider device idle (default: 300) | Block device |
| `--max-idle-skip <sec>` | Max seconds to skip due to idle (default: 86400) | Block device |
| `--btrfs-scrub-days <n>` | Days between btrfs scrub runs (default: 30) | Btrfs |

### Error Thresholds (`--smart`)

For block devices, thresholds are SMART attributes:

```bash
--smart "pending=0,realloc=5,uncorrectable=0,reported_uncorrect=0"
```

For btrfs, thresholds are scrub error counters:

```bash
--smart "read_errors=0,csum_errors=0,verify_errors=0,super_errors=0,uncorrectable_errors=0"
```

### Examples

NVMe drive with a higher temperature limit:

```bash
sudo ./diskmon.py /dev/nvme0n1 --kuma-url http://kuma.local/push/XXXXX --temp-threshold 60
```

Btrfs filesystem with weekly scrubs and tolerance for corrected checksum errors:

```bash
sudo ./diskmon.py /media/backups --kuma-url http://kuma.local/push/XXXXX \
  --btrfs-scrub-days 7 --smart "csum_errors=5"
```

---

## Configuration File

Diskmon supports a config file (default: `~/.config/diskmon.conf`).

### Block Device Example

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

### Btrfs Example

```ini
[diskmon]
usage_threshold = 85
btrfs_scrub_interval = 14
log_file = /var/log/diskmon.log

[btrfs_scrub_thresholds]
read_errors = 0
csum_errors = 0
verify_errors = 0
super_errors = 0
uncorrectable_errors = 0
```

---

## Integration with Uptime Kuma

Diskmon pushes results directly to Kuma's HTTP(s) push monitors, reporting:

* Status: `up` or `down`
* Message: latest health summary
* Ping: duration of the health check in ms

---

## Disclaimer

Like most of the code on Github these days, this script was **completely vibe coded** (and so was the README as I'm sure you can tell).
While it works in practice, **human eyes have not reviewed it in depth**.

Use **at your own risk**.
