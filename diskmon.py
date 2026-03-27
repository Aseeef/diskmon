#!/usr/bin/env python3
"""
Diskmon - Linux Disk Health Monitoring Tool
Production-ready implementation with optimal balance of robustness, maintainability, and elegance.
"""

import os
import sys
import argparse
import configparser
import json
import logging
import logging.handlers
import stat
import subprocess
import time
import signal
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum, auto

# Dependency check with helpful error message
REQUIRED_PACKAGES = ['requests', 'filelock', 'psutil']
missing_packages = []

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"ERROR: Missing required Python packages: {', '.join(missing_packages)}", file=sys.stderr)
    print(f"Please install them using: pip3 install {' '.join(missing_packages)}", file=sys.stderr)
    sys.exit(1)

import requests
import filelock
import psutil


# ============================================================================
# Constants and Enums
# ============================================================================

class OutputFormat(Enum):
    """Output format options"""
    HUMAN = 'human'
    JSON = 'json'
    KUMA_ONLY = 'kuma-only'


class HealthStatus(Enum):
    """Health check status"""
    UP = 'up'
    DOWN = 'down'


# SMART attribute IDs and their friendly names
SMART_ATTRIBUTES = {
    5: ('realloc', 'Reallocated_Sector_Ct'),
    187: ('reported_uncorrect', 'Reported_Uncorrect'),
    197: ('pending', 'Current_Pending_Sector'),
    198: ('uncorrectable', 'Offline_Uncorrectable'),
}

# Btrfs scrub error fields that are checked against thresholds
BTRFS_SCRUB_ERROR_FIELDS = {
    'read_errors': 'Read Errors',
    'csum_errors': 'Checksum Errors',
    'verify_errors': 'Verify Errors',
    'super_errors': 'Superblock Errors',
    'uncorrectable_errors': 'Uncorrectable Errors',
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DiskmonConfig:
    """Configuration for diskmon"""
    device: str
    kuma_url: str
    config_file: str = "~/.config/diskmon.conf"
    
    # Thresholds
    usage_threshold: float = 90.0
    temp_threshold: int = 50
    smart_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'pending': 0,
        'realloc': 0,
        'uncorrectable': 0,
        'reported_uncorrect': 0
    })
    
    # Test scheduling (days)
    short_test_interval: int = 7
    long_test_interval: int = 30
    badblocks_interval: Optional[int] = None  # Off by default
    btrfs_scrub_interval: int = 30  # days
    
    # Idle detection
    idle_threshold: int = 300  # seconds
    max_idle_skip: int = 86400  # 24 hours
    
    # Behavior
    require_device_present: bool = True
    log_file: str = "/var/log/diskmon.log"
    output_format: OutputFormat = OutputFormat.HUMAN
    verbose: bool = False
    color_output: bool = True  # New: color support for human format
    
    # Btrfs mode
    is_btrfs: bool = False
    btrfs_scrub_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'read_errors': 0,
        'csum_errors': 0,
        'verify_errors': 0,
        'super_errors': 0,
        'uncorrectable_errors': 0,
    })


@dataclass
class DeviceState:
    """Persistent state for a monitored device"""
    device_path: str
    last_check_ts: Optional[float] = None
    last_short_test_scheduled_ts: Optional[float] = None
    last_long_test_scheduled_ts: Optional[float] = None
    last_badblocks_scheduled_ts: Optional[float] = None
    last_btrfs_scrub_scheduled_ts: Optional[float] = None
    last_successful_smart_read_ts: Optional[float] = None
    last_io_ticks: Optional[int] = None
    
    # Cached status for idle reporting
    last_known_healthy: bool = True
    last_known_message: str = "Initial state"
    
    # Additional tracking
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0


@dataclass
class SMARTData:
    """Parsed SMART data from smartctl"""
    device_type: Optional[str] = None
    smart_status_passed: bool = True
    temperature: Optional[int] = None
    power_on_hours: Optional[int] = None
    attributes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    self_tests: List[Dict[str, Any]] = field(default_factory=list)

    # Store the raw NVMe log for direct evaluation
    nvme_health_log: Optional[Dict[str, Any]] = None

    def get_attribute_raw_value(self, attr_id: int) -> int:
        """Get raw value of a SMART attribute (for ATA devices)"""
        attr = self.attributes.get(attr_id, {})
        return attr.get('raw', {}).get('value', 0)

    def get_latest_self_test(self) -> Optional[Dict[str, Any]]:
        """Get most recent self-test result (for ATA devices)"""
        return self.self_tests[0] if self.self_tests else None


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    is_healthy: bool
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class BtrfsScrubStatus(Enum):
    """Btrfs scrub operation status"""
    FINISHED = auto()
    RUNNING = auto()
    ABORTED = auto()
    NO_HISTORY = auto()
    FAILED = auto()


@dataclass
class BtrfsScrubData:
    """Parsed btrfs scrub status data"""
    started: Optional[str] = None
    status: Optional[str] = None
    duration: Optional[str] = None
    errors: Dict[str, int] = field(default_factory=dict)
    data_bytes_scrubbed: int = 0
    tree_bytes_scrubbed: int = 0
    corrected_errors: int = 0


# ============================================================================
# Core Components
# ============================================================================

class PrivilegeChecker:
    """Ensures appropriate privileges"""
    
    @staticmethod
    def check():
        """Verify running with sudo privileges"""
        if os.geteuid() != 0:
            print("ERROR: diskmon must be run with sudo privileges", file=sys.stderr)
            print("Usage: sudo diskmon /dev/sdX --kuma-url URL", file=sys.stderr)
            sys.exit(1)


class ConfigManager:
    """Handles configuration loading and parsing"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = DiskmonConfig(
            device=args.device,
            kuma_url=args.kuma_url
        )
    
    def load(self) -> DiskmonConfig:
        """Load configuration with proper precedence"""
        self._detect_mode()
        self._load_from_file()
        self._apply_cli_overrides()
        self._validate_config()
        return self.config
    
    def _detect_mode(self):
        """Detect if the target is a btrfs mount path vs a block device.
        
        A directory path implies btrfs mode; the actual filesystem validation
        happens when `btrfs scrub status` is called during the health check.
        """
        device_path = Path(self.config.device)
        if not device_path.is_dir():
            return
        
        self.config.is_btrfs = True
        self.config.device = str(device_path.resolve())
    
    def _load_from_file(self):
        """Load configuration from file if it exists"""
        config_path = Path(os.path.expanduser(
            self.args.config or self.config.config_file
        ))
        
        if not config_path.exists():
            logging.debug(f"Config file {config_path} not found, using defaults")
            return
        
        parser = configparser.ConfigParser()
        try:
            parser.read(config_path)
            
            # Load main section
            if 'diskmon' in parser:
                section = parser['diskmon']
                self._load_section_values(section)
            
            # Load SMART thresholds
            if 'smart_thresholds' in parser:
                section = parser['smart_thresholds']
                self._load_smart_thresholds(section)
            
            # Load btrfs scrub thresholds
            if 'btrfs_scrub_thresholds' in parser:
                section = parser['btrfs_scrub_thresholds']
                self._load_btrfs_scrub_thresholds(section)
                
        except Exception as e:
            logging.warning(f"Error parsing config file {config_path}: {e}")
    
    def _load_section_values(self, section):
        """Load values from a config section"""
        self.config.usage_threshold = section.getfloat(
            'usage_threshold', self.config.usage_threshold)
        self.config.temp_threshold = section.getint(
            'temp_threshold', self.config.temp_threshold)
        self.config.short_test_interval = section.getint(
            'short_test_interval', self.config.short_test_interval)
        self.config.long_test_interval = section.getint(
            'long_test_interval', self.config.long_test_interval)
        
        # Badblocks is optional
        if 'badblocks_interval' in section:
            self.config.badblocks_interval = section.getint('badblocks_interval')
        
        self.config.btrfs_scrub_interval = section.getint(
            'btrfs_scrub_interval', self.config.btrfs_scrub_interval)
        
        self.config.idle_threshold = section.getint(
            'idle_threshold', self.config.idle_threshold)
        self.config.max_idle_skip = section.getint(
            'max_idle_skip', self.config.max_idle_skip)
        self.config.log_file = section.get(
            'log_file', self.config.log_file)
    
    def _load_smart_thresholds(self, section):
        """Load SMART thresholds from config section"""
        for key in self.config.smart_thresholds:
            if key in section:
                self.config.smart_thresholds[key] = section.getint(key)
    
    def _load_btrfs_scrub_thresholds(self, section):
        """Load btrfs scrub error thresholds from config section"""
        for key in self.config.btrfs_scrub_thresholds:
            if key in section:
                self.config.btrfs_scrub_thresholds[key] = section.getint(key)
    
    def _apply_cli_overrides(self):
        """Apply command-line argument overrides"""
        # Simple overrides
        if self.args.usage_threshold is not None:
            self.config.usage_threshold = self.args.usage_threshold
        if self.args.temp_threshold is not None:
            self.config.temp_threshold = self.args.temp_threshold
        if self.args.short_test_days is not None:
            self.config.short_test_interval = self.args.short_test_days
        if self.args.long_test_days is not None:
            self.config.long_test_interval = self.args.long_test_days
        if self.args.badblocks_days is not None:
            self.config.badblocks_interval = self.args.badblocks_days
        if self.args.btrfs_scrub_days is not None:
            self.config.btrfs_scrub_interval = self.args.btrfs_scrub_days
        if self.args.idle_threshold is not None:
            self.config.idle_threshold = self.args.idle_threshold
        if self.args.max_idle_skip is not None:
            self.config.max_idle_skip = self.args.max_idle_skip
        if self.args.log_file:
            self.config.log_file = self.args.log_file
        
        # Boolean flags
        self.config.require_device_present = self.args.require_device
        self.config.verbose = self.args.verbose
        self.config.color_output = not self.args.no_color
        
        # Output format
        self.config.output_format = OutputFormat(self.args.output)
        
        # Parse SMART thresholds
        if self.args.smart:
            self._parse_smart_cli()
    
    def _parse_smart_cli(self):
        """Parse error thresholds from CLI argument (SMART or btrfs scrub)"""
        thresholds = (self.config.btrfs_scrub_thresholds
                      if self.config.is_btrfs
                      else self.config.smart_thresholds)
        try:
            for item in self.args.smart.split(','):
                key, value = item.strip().split('=')
                key = key.strip()
                if key in thresholds:
                    thresholds[key] = int(value.strip())
                else:
                    valid_keys = ', '.join(thresholds.keys())
                    logging.warning(f"Unknown threshold key: {key} (valid: {valid_keys})")
        except ValueError:
            logging.error(f"Invalid --smart format: {self.args.smart}")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.config.usage_threshold < 0 or self.config.usage_threshold > 100:
            logging.error("Usage threshold must be between 0 and 100")
            sys.exit(1)
        
        if self.config.temp_threshold < 0:
            logging.error("Temperature threshold must be positive")
            sys.exit(1)
        
        if self.config.is_btrfs:
            unsupported = []
            if self.args.temp_threshold is not None:
                unsupported.append('--temp-threshold')
            if self.args.short_test_days is not None:
                unsupported.append('--short-test-days')
            if self.args.long_test_days is not None:
                unsupported.append('--long-test-days')
            if self.args.idle_threshold is not None:
                unsupported.append('--idle-threshold')
            if self.args.max_idle_skip is not None:
                unsupported.append('--max-idle-skip')
            if self.args.badblocks_days is not None:
                unsupported.append('--badblocks-days')
            if unsupported:
                print(f"ERROR: Options not supported for btrfs: {', '.join(unsupported)}", file=sys.stderr)
                sys.exit(1)
        elif self.args.btrfs_scrub_days is not None:
            print("ERROR: --btrfs-scrub-days is only valid for btrfs mount paths", file=sys.stderr)
            sys.exit(1)
    
    @staticmethod
    def get_cli_parser() -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Diskmon - Linux Disk Health Monitoring Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Required arguments
        parser.add_argument('device', 
                          help='Block device (e.g., /dev/sda) or btrfs mount path (e.g., /media/backups)')
        parser.add_argument('--kuma-url', required=True,
                          help='Uptime Kuma push URL')
        
        # Configuration
        parser.add_argument('--config',
                          help='Configuration file path')
        
        # Thresholds
        parser.add_argument('--usage-threshold', type=float,
                          help='Filesystem usage threshold %% (default: 90)')
        parser.add_argument('--temp-threshold', type=int,
                          help='Temperature threshold °C (default: 50)')
        parser.add_argument('--smart',
                          help='Error thresholds: SMART "pending=N,realloc=N,..." '
                               'or btrfs "read_errors=N,csum_errors=N,..."')
        
        # Test scheduling
        parser.add_argument('--short-test-days', type=int,
                          help='Days between short SMART tests (default: 7)')
        parser.add_argument('--long-test-days', type=int,
                          help='Days between long SMART tests (default: 30)')
        parser.add_argument('--badblocks-days', type=int,
                          help='Days between bad block scans (optional)')
        parser.add_argument('--btrfs-scrub-days', type=int,
                          help='Days between btrfs scrub runs (default: 30, btrfs only)')
        
        # Idle detection
        parser.add_argument('--idle-threshold', type=int,
                          help='Seconds to consider device idle (default: 300)')
        parser.add_argument('--max-idle-skip', type=int,
                          help='Max seconds to skip due to idle (default: 86400)')
        
        # Behavior
        parser.add_argument('--no-require-device', dest='require_device',
                          action='store_false',
                          help="Don't fail if device doesn't exist")
        parser.add_argument('--log-file', 
                          help='Log file path (default: /var/log/diskmon.log)')
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
        parser.add_argument('--no-color', action='store_true',
                          help='Disable colored output')
        parser.add_argument('--output', 
                          choices=['human', 'json', 'kuma-only'],
                          default='human',
                          help='Output format (default: human)')
        
        return parser


class StateManager:
    """Manages persistent state with multi-process safety"""
    
    def __init__(self, device_path: str):
        path = Path(device_path)
        if path.is_dir():
            self.device_name = str(path.resolve()).strip('/').replace('/', '_')
        else:
            self.device_name = path.name
        self.state_dir = Path("/var/lib/diskmon/devices")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / f"{self.device_name}.json"
        self.lock_file = self.state_dir / f"{self.device_name}.lock"
        self.lock = filelock.FileLock(str(self.lock_file), timeout=30)
    
    def load(self) -> DeviceState:
        """Load state from file"""
        if not self.state_file.exists():
            return DeviceState(device_path=self.device_name)
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                return DeviceState(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Corrupted state file, starting fresh: {e}")
            return DeviceState(device_path=self.device_name)
    
    def save(self, state: DeviceState):
        """Save state atomically"""
        temp_file = self.state_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(asdict(state), f, indent=2)
            temp_file.replace(self.state_file)
        except IOError as e:
            logging.error(f"Failed to save state: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def is_test_due(self, last_ts: Optional[float], interval_days: int) -> bool:
        """Check if a test is due based on last timestamp"""
        if last_ts is None:
            return True
        elapsed = time.time() - last_ts
        return elapsed >= (interval_days * 86400)


class DeviceManager:
    """Handles device detection and power state"""
    
    def exists(self, device_path: str) -> bool:
        """Check if device exists and is a block device"""
        try:
            path = Path(device_path)
            if not path.exists():
                return False
            return stat.S_ISBLK(path.stat().st_mode)
        except OSError:
            return False
    
    def get_io_statistics(self, device_path: str) -> Optional[int]:
        """Get I/O ticks from /proc/diskstats"""
        device_name = Path(device_path).name
        
        try:
            with open('/proc/diskstats', 'r') as f:
                for line in f:
                    fields = line.split()
                    if len(fields) >= 14 and fields[2] == device_name:
                        return int(fields[12])  # io_ticks
        except (IOError, ValueError) as e:
            logging.debug(f"Could not read diskstats: {e}")
        
        return None
    
    def is_idle(self, config: DiskmonConfig, state: DeviceState) -> Tuple[bool, str]:
        """Check if device should be skipped due to idle state"""
        current_ticks = self.get_io_statistics(config.device)
        
        if current_ticks is None:
            return False, "Could not determine I/O activity"
        
        # Check if I/O has occurred since last check
        if state.last_io_ticks is not None:
            if current_ticks == state.last_io_ticks:
                # No I/O since last check
                time_since_check = time.time() - (state.last_check_ts or 0)
                
                if time_since_check < config.idle_threshold:
                    state.last_io_ticks = current_ticks
                    return False, "Device active recently"
                
                # Device is idle, check if we've been skipping too long
                time_since_smart = time.time() - (state.last_successful_smart_read_ts or 0)
                if time_since_smart > config.max_idle_skip:
                    state.last_io_ticks = current_ticks
                    return False, f"Forcing check after {time_since_smart:.0f}s"
                
                return True, f"Device idle for {time_since_check:.0f}s"
        
        # First check or device was active
        state.last_io_ticks = current_ticks
        return False, "Device is active"


class FilesystemMonitor:
    """Monitors filesystem usage"""
    
    def get_usage(self, device_path: str) -> Optional[float]:
        """Get filesystem usage percentage"""
        try:
            if Path(device_path).is_dir():
                usage = psutil.disk_usage(device_path)
                return usage.percent
            
            # Check direct device mount
            for partition in psutil.disk_partitions(all=False):
                if partition.device == device_path:
                    usage = psutil.disk_usage(partition.mountpoint)
                    return usage.percent
            
            # Check if it's a parent device with partitions
            device_name = Path(device_path).name
            for partition in psutil.disk_partitions(all=False):
                if device_name in partition.device:
                    usage = psutil.disk_usage(partition.mountpoint)
                    return usage.percent
                    
        except Exception as e:
            logging.debug(f"Could not get disk usage: {e}")
        
        return None

class SmartStatus(Enum):
    SUCCESS = auto()
    SLEEPING = auto()
    FAILED = auto()
    TEST_IN_PROGRESS = auto()

class SMARTHandler:
    """Handles SMART operations and evaluation"""
    
    def get_data(self, device_path: str) -> Tuple[SmartStatus, Optional[SMARTData]]:
        """Get SMART data and return a status indicating the outcome."""
        try:
            # A single, comprehensive call is cleaner and more efficient.
            result = self._run_smartctl(['-j', '-a', device_path], timeout=60)

            # Check for sleeping device first (returncode 2).
            if result.returncode == 2:
                return SmartStatus.SLEEPING, None

            # Check for critical command failures (bitmask: 1=open fail, 2=cmd fail).
            if (result.returncode & 3) != 0:
                logging.error(f"smartctl command failed with critical error code {result.returncode}: {result.stderr}")
                return SmartStatus.FAILED, None

            # Non-critical errors (like checksum warnings) can proceed.
            if "invalid SMART checksum" in result.stderr:
                logging.warning("Proceeding with SMART data parsing despite checksum warning.")

            parsed_data = self._parse_json(result.stdout)
            if parsed_data is None:
                return SmartStatus.FAILED, None

            # Now, determine the status from the complete data. This check is primarily for ATA drives.
            data_dict = json.loads(result.stdout)
            if "in progress" in data_dict.get("self_test_status", {}).get("string", "").lower():
                 logging.info(f"SMART self-test is in progress.")
                 return SmartStatus.TEST_IN_PROGRESS, parsed_data

            return SmartStatus.SUCCESS, parsed_data

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"SMART data collection failed: {e}")
            return SmartStatus.FAILED, None
    
    def _run_smartctl(self, args: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Run smartctl command"""
        cmd = ['smartctl'] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def _parse_json(self, json_str: str) -> Optional[SMARTData]:
        """Parse smartctl JSON output, supporting both ATA and NVMe devices."""
        try:
            data = json.loads(json_str)
            smart_data = SMARTData()

            # --- Common Fields ---
            smart_data.smart_status_passed = data.get('smart_status', {}).get('passed', False)
            smart_data.device_type = data.get('device', {}).get('type')

            # Use top-level temperature and power_on_time as they are more consistent
            smart_data.temperature = data.get('temperature', {}).get('current')
            smart_data.power_on_hours = data.get('power_on_time', {}).get('hours')

            # --- Device-Specific Parsing ---
            if smart_data.device_type == 'nvme':
                # For NVMe, the most important data is in the health information log
                smart_data.nvme_health_log = data.get('nvme_smart_health_information_log')

                # If top-level temp wasn't found, try inside the log (fallback)
                if smart_data.temperature is None and smart_data.nvme_health_log:
                    smart_data.temperature = smart_data.nvme_health_log.get('temperature')

            else: # Default to ATA/SATA parsing
                smart_data.device_type = 'ata' # Be explicit for clarity
                # Attributes
                for attr in data.get('ata_smart_attributes', {}).get('table', []):
                    if 'id' in attr:
                        smart_data.attributes[attr['id']] = attr

                # Self-test log (only relevant for ATA)
                log_data = data.get('ata_smart_self_test_log', {})
                smart_data.self_tests = log_data.get('standard', {}).get('table', [])

            return smart_data

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse SMART JSON: {e}")
            return None

    def evaluate(self, data: SMARTData, config: DiskmonConfig, is_test_in_progress: bool = False) -> Tuple[bool, List[str]]:
        """Evaluate SMART data against thresholds based on device type."""
        if data.device_type == 'nvme':
            return self._evaluate_nvme(data, config)
        else: # Default to ATA
            return self._evaluate_ata(data, config, is_test_in_progress)

    def _evaluate_nvme(self, data: SMARTData, config: DiskmonConfig) -> Tuple[bool, List[str]]:
        """Evaluate NVMe SMART data against thresholds."""
        issues = []

        # 1. Overall health status
        if not data.smart_status_passed:
            issues.append("SMART overall health: FAILED")

        # 2. Temperature
        if data.temperature and data.temperature > config.temp_threshold:
            issues.append(f"Temperature: {data.temperature}°C > {config.temp_threshold}°C")

        log = data.nvme_health_log
        if not log:
            # This check is crucial; if the log is missing, we can't proceed.
            if not issues: # Only add this if no other issue was found
                 issues.append("NVMe SMART health log not found in smartctl output")
            return False, issues

        # 3. Critical Warning Flags (Bitmap)
        critical_warning = log.get('critical_warning', 0)
        if critical_warning > 0:
            warnings = []
            if critical_warning & 0x1: warnings.append("Available spare is below threshold")
            if critical_warning & 0x2: warnings.append("Temperature has exceeded threshold")
            if critical_warning & 0x4: warnings.append("NVM subsystem reliability is degraded")
            if critical_warning & 0x8: warnings.append("Media is in read-only mode")
            if critical_warning & 0x10: warnings.append("Volatile memory backup device has failed")
            issues.append(f"Critical Warning flags set: {', '.join(warnings) or 'Unknown'}")

        # 4. Media and Data Integrity Errors (maps to your 'uncorrectable' threshold)
        media_errors = log.get('media_errors', 0)
        threshold = config.smart_thresholds.get('uncorrectable', 0)
        if media_errors > threshold:
            issues.append(f"Media Errors: {media_errors} > {threshold}")

        # Note: NVMe self-tests are not checked as they are often unsupported or logged differently.
        return len(issues) == 0, issues

    def _evaluate_ata(self, data: SMARTData, config: DiskmonConfig, is_test_in_progress: bool) -> Tuple[bool, List[str]]:
        """Evaluate ATA SMART data against thresholds (original logic)."""
        issues = []

        # Overall SMART status
        if not data.smart_status_passed:
            issues.append("SMART overall health: FAILED")

        # Check critical attributes
        for attr_id, (key, name) in SMART_ATTRIBUTES.items():
            raw_value = data.get_attribute_raw_value(attr_id)
            threshold = config.smart_thresholds.get(key, 0)

            if raw_value > threshold:
                issues.append(f"{name}: {raw_value} > {threshold}")

        # Temperature
        if data.temperature and data.temperature > config.temp_threshold:
            issues.append(f"Temperature: {data.temperature}°C > {config.temp_threshold}°C")

        # Latest self-test (SKIP THIS CHECK IF A TEST IS CURRENTLY RUNNING)
        if not is_test_in_progress:
            latest_test = data.get_latest_self_test()
            if latest_test:
                status = latest_test.get('status', {})
                if not status.get('passed', True):
                    issues.append(f"Self-test failed: {status.get('string', 'Unknown')}")

        return len(issues) == 0, issues
    
    def schedule_test(self, device_path: str, test_type: str) -> bool:
        """Schedule a SMART self-test"""
        if test_type not in ['short', 'long']:
            return False
        
        try:
            result = self._run_smartctl(['-t', test_type, device_path], timeout=30)
            success = result.returncode == 0
            
            if success:
                logging.info(f"Scheduled {test_type} test on {device_path}")
            else:
                logging.error(f"Failed to schedule {test_type} test: {result.stderr}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error scheduling {test_type} test: {e}")
            return False
    
    def schedule_badblocks(self, device_path: str) -> bool:
        """Schedule a background badblocks scan"""
        log_file = f"/var/log/diskmon_badblocks_{Path(device_path).name}.log"
        
        try:
            # Use subprocess.Popen for true background execution
            device_name = os.path.basename(device_path)
            log_path = f"/var/log/diskmon_badblocks_{device_name}.log"
            cmd = f"nohup badblocks -sv {device_path} > {log_path} 2>&1 &"
            subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setpgrp,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logging.info(f"Started badblocks scan, output: {log_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start badblocks: {e}")
            return False


class BtrfsScrubHandler:
    """Handles btrfs scrub operations and evaluation"""
    
    def get_status(self, mount_path: str) -> Tuple[BtrfsScrubStatus, Optional[BtrfsScrubData]]:
        """Get btrfs scrub status for a mount path"""
        try:
            result = subprocess.run(
                ['btrfs', 'scrub', 'status', '-R', mount_path],
                capture_output=True, text=True, timeout=30
            )
            
            combined = result.stdout + result.stderr
            if 'no stats available' in combined.lower():
                return BtrfsScrubStatus.NO_HISTORY, None
            
            if result.returncode != 0:
                logging.error(f"btrfs scrub status failed (rc={result.returncode}): {result.stderr}")
                return BtrfsScrubStatus.FAILED, None
            
            return self._parse_status(result.stdout)
            
        except subprocess.TimeoutExpired:
            logging.error("btrfs scrub status timed out")
            return BtrfsScrubStatus.FAILED, None
        except FileNotFoundError:
            logging.error("btrfs command not found")
            return BtrfsScrubStatus.FAILED, None
    
    def _parse_status(self, output: str) -> Tuple[BtrfsScrubStatus, Optional[BtrfsScrubData]]:
        """Parse btrfs scrub status -R output"""
        data = BtrfsScrubData()
        
        for line in output.splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            key, _, value = line.partition(':')
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'scrub started':
                data.started = value
            elif key == 'status':
                data.status = value.lower()
            elif key == 'duration':
                data.duration = value
            elif key == 'data_bytes_scrubbed':
                data.data_bytes_scrubbed = int(value)
            elif key == 'tree_bytes_scrubbed':
                data.tree_bytes_scrubbed = int(value)
            elif key == 'corrected_errors':
                data.corrected_errors = int(value)
            elif key in BTRFS_SCRUB_ERROR_FIELDS:
                data.errors[key] = int(value)
        
        if data.status is None:
            logging.error("Could not parse scrub status from btrfs output")
            return BtrfsScrubStatus.FAILED, None
        
        status_map = {
            'finished': BtrfsScrubStatus.FINISHED,
            'running': BtrfsScrubStatus.RUNNING,
            'aborted': BtrfsScrubStatus.ABORTED,
        }
        scrub_status = status_map.get(data.status, BtrfsScrubStatus.FAILED)
        return scrub_status, data
    
    def evaluate(self, data: BtrfsScrubData, config: DiskmonConfig) -> Tuple[bool, List[str]]:
        """Evaluate btrfs scrub data against thresholds"""
        issues = []
        
        for field_name, display_name in BTRFS_SCRUB_ERROR_FIELDS.items():
            count = data.errors.get(field_name, 0)
            threshold = config.btrfs_scrub_thresholds.get(field_name, 0)
            if count > threshold:
                issues.append(f"{display_name}: {count} > {threshold}")
        
        return len(issues) == 0, issues
    
    def start_scrub(self, mount_path: str) -> bool:
        """Start a btrfs scrub in the background.
        
        Uses Popen to avoid blocking — the kernel runs the scrub
        asynchronously and we only need to kick it off.
        """
        try:
            proc = subprocess.Popen(
                ['btrfs', 'scrub', 'start', mount_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True
            )
            try:
                stdout, stderr = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                logging.info(f"btrfs scrub start still running after 60s, detaching (scrub continues in kernel)")
                proc.stdout.close()
                proc.stderr.close()
                return True
            
            combined = (stdout.decode() + stderr.decode()).lower()
            if proc.returncode == 0 or 'already running' in combined:
                logging.info(f"Started btrfs scrub on {mount_path}")
                return True
            
            logging.error(f"Failed to start btrfs scrub (rc={proc.returncode}): {stderr.decode().strip()}")
            return False
            
        except FileNotFoundError:
            logging.error("btrfs command not found")
            return False
        except Exception as e:
            logging.error(f"Error starting btrfs scrub: {e}")
            return False


class HealthChecker:
    """Coordinates health checks"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.device_manager = DeviceManager()
        self.fs_monitor = FilesystemMonitor()
        self.smart_handler = SMARTHandler()
        self.btrfs_handler = BtrfsScrubHandler()
        
    def check(self, config: DiskmonConfig, state: DeviceState) -> HealthCheckResult:
        """Perform complete health check, dispatching to btrfs or SMART path."""
        start_time = time.time()
        
        if config.is_btrfs:
            return self._check_btrfs(config, state, start_time)
        
        return self._check_block_device(config, state, start_time)
    
    def _check_block_device(self, config: DiskmonConfig, state: DeviceState, start_time: float) -> HealthCheckResult:
        """Health check for block devices using SMART."""
        issues = []
        details = {}

        # 1. Idle check remains first.
        is_idle, idle_reason = self.device_manager.is_idle(config, state)
        if is_idle:
            details['idle'] = True
            details['idle_reason'] = idle_reason
            return self._result(state.last_known_healthy, f"Idle - cached: {state.last_known_message}", start_time, details)

        # --- DISK IS ACTIVE ---
        if not self.device_manager.exists(config.device):
            if config.require_device_present:
                return self._result(False, "Device not found", start_time, details)
            else:
                return self._result(True, "Device not present (ignored)", start_time, details)

        usage = self.fs_monitor.get_usage(config.device)
        if usage is not None:
            details['usage_percent'] = usage
            if usage > config.usage_threshold:
                issues.append(f"Disk usage: {usage:.1f}% > {config.usage_threshold}%")

        smart_status, smart_data = self.smart_handler.get_data(config.device)
        
        if smart_status == SmartStatus.SLEEPING:
            reason = "sleeping"
            details['smart_skipped'] = reason
            return self._result(state.last_known_healthy, f"Check deferred ({reason}) - cached: {state.last_known_message}", start_time, details)

        elif smart_status == SmartStatus.FAILED:
            issues.append("Failed to execute or parse smartctl command")

        elif smart_data: # Handles SUCCESS and TEST_IN_PROGRESS
            # HOLE #1 FIX: Populate all available details, even in best-effort mode.
            details['temperature'] = smart_data.temperature
            details['power_on_hours'] = smart_data.power_on_hours
            details['smart_passed'] = smart_data.smart_status_passed

            is_healthy, smart_issues = self.smart_handler.evaluate(smart_data, config, is_test_in_progress=(smart_status == SmartStatus.TEST_IN_PROGRESS))
            issues.extend(smart_issues)
            
            if smart_status == SmartStatus.SUCCESS:
                state.last_successful_smart_read_ts = time.time()
                self._schedule_tests(config, state, smart_data)

        # Build final result
        is_healthy = len(issues) == 0
        
        # HOLE #2 FIX: Add context to failure messages when a test is in progress.
        if not is_healthy and smart_status == SmartStatus.TEST_IN_PROGRESS:
            message = f"(Test in Progress) {'; '.join(issues)}"
        elif not is_healthy:
            message = "; ".join(issues)
        else: # is_healthy
            message = self._build_success_message(details, smart_status)
        
        # Update state and return
        state.last_known_healthy = is_healthy
        state.last_known_message = message
        state.total_checks += 1
        if not is_healthy:
            state.consecutive_failures += 1
            state.total_failures += 1
        else:
            state.consecutive_failures = 0
        
        return self._result(is_healthy, message, start_time, details)

    def _build_success_message(self, details: Dict[str, Any], smart_status: SmartStatus) -> str:
        """Helper to build the rich success message."""
        status_text = "OK (Test in Progress)" if smart_status == SmartStatus.TEST_IN_PROGRESS else "OK"
        parts = [status_text]
        usage = details.get('usage_percent')
        temp = details.get('temperature')
        
        if usage is not None:
            parts.append(f"Usage: {usage:.1f}%")
        if temp is not None:
            parts.append(f"Temp: {temp}°C")
        
        return " | ".join(parts)
    
    def _check_btrfs(self, config: DiskmonConfig, state: DeviceState, start_time: float) -> HealthCheckResult:
        """Health check for btrfs mount paths using scrub data."""
        issues = []
        details = {}
        
        if not Path(config.device).is_dir():
            return self._result(False, "Btrfs mount path not found", start_time, details)
        
        usage = self.fs_monitor.get_usage(config.device)
        if usage is not None:
            details['usage_percent'] = usage
            if usage > config.usage_threshold:
                issues.append(f"Disk usage: {usage:.1f}% > {config.usage_threshold}%")
        
        scrub_status, scrub_data = self.btrfs_handler.get_status(config.device)
        details['btrfs_scrub_status'] = scrub_status.name.lower()
        
        if scrub_status == BtrfsScrubStatus.FAILED:
            issues.append("Failed to get btrfs scrub status")
        
        elif scrub_status == BtrfsScrubStatus.NO_HISTORY:
            details['btrfs_scrub_status'] = 'no_history'
        
        elif scrub_data:
            if scrub_data.started:
                details['btrfs_scrub_started'] = scrub_data.started
            if scrub_data.duration:
                details['btrfs_scrub_duration'] = scrub_data.duration
            if scrub_data.data_bytes_scrubbed:
                details['btrfs_data_scrubbed_gb'] = round(scrub_data.data_bytes_scrubbed / (1024**3), 1)
            if scrub_data.corrected_errors:
                details['btrfs_corrected_errors'] = scrub_data.corrected_errors
            
            is_ok, scrub_issues = self.btrfs_handler.evaluate(scrub_data, config)
            issues.extend(scrub_issues)
            
            if scrub_status == BtrfsScrubStatus.ABORTED:
                issues.append("Last btrfs scrub was aborted")
        
        if scrub_status not in (BtrfsScrubStatus.RUNNING, BtrfsScrubStatus.FAILED):
            self._schedule_btrfs_scrub(config, state)
        
        is_healthy = len(issues) == 0
        
        if not is_healthy and scrub_status == BtrfsScrubStatus.RUNNING:
            message = f"(Scrub Running) {'; '.join(issues)}"
        elif not is_healthy:
            message = "; ".join(issues)
        else:
            message = self._build_btrfs_success_message(details, scrub_status)
        
        state.last_known_healthy = is_healthy
        state.last_known_message = message
        state.total_checks += 1
        if not is_healthy:
            state.consecutive_failures += 1
            state.total_failures += 1
        else:
            state.consecutive_failures = 0
        
        return self._result(is_healthy, message, start_time, details)
    
    def _build_btrfs_success_message(self, details: Dict[str, Any], scrub_status: BtrfsScrubStatus) -> str:
        """Build success message for btrfs checks."""
        if scrub_status == BtrfsScrubStatus.RUNNING:
            status_text = "OK (Scrub Running)"
        elif scrub_status == BtrfsScrubStatus.NO_HISTORY:
            status_text = "OK (No Scrub History)"
        else:
            status_text = "OK"
        
        parts = [status_text]
        usage = details.get('usage_percent')
        if usage is not None:
            parts.append(f"Usage: {usage:.1f}%")
        
        scrub_started = details.get('btrfs_scrub_started')
        if scrub_started:
            parts.append(f"Last Scrub: {scrub_started}")
        
        return " | ".join(parts)
    
    def _schedule_btrfs_scrub(self, config: DiskmonConfig, state: DeviceState):
        """Schedule a btrfs scrub if due"""
        if not self.state_manager.is_test_due(
            state.last_btrfs_scrub_scheduled_ts,
            config.btrfs_scrub_interval
        ):
            return
        
        if self.btrfs_handler.start_scrub(config.device):
            state.last_btrfs_scrub_scheduled_ts = time.time()
    
    def _schedule_tests(self, config: DiskmonConfig, state: DeviceState, smart_data: SMARTData):
        """Schedule tests if due"""
        now = time.time()
        
        if smart_data.device_type == 'nvme':
            logging.info("Skipping ATA/SATA self-test scheduling for NVMe device.")
            # We can also update the timestamp here to prevent re-checking every time
            # This signifies we've "handled" the schedule check for this interval.
            if state.last_short_test_scheduled_ts is None:
                state.last_short_test_scheduled_ts = now
            if state.last_long_test_scheduled_ts is None:
                state.last_long_test_scheduled_ts = now
            return

        # Short test (This code will now only run for non-NVMe drives)
        if self.state_manager.is_test_due(
            state.last_short_test_scheduled_ts,
            config.short_test_interval
        ):
            if self.smart_handler.schedule_test(config.device, 'short'):
                state.last_short_test_scheduled_ts = now

        # Long test
        if self.state_manager.is_test_due(
            state.last_long_test_scheduled_ts,
            config.long_test_interval
        ):
            if self.smart_handler.schedule_test(config.device, 'long'):
                state.last_long_test_scheduled_ts = now

        # Badblocks (if enabled)
        if config.badblocks_interval and self.state_manager.is_test_due(
            state.last_badblocks_scheduled_ts,
            config.badblocks_interval
        ):
            if self.smart_handler.schedule_badblocks(config.device):
                state.last_badblocks_scheduled_ts = now
    
    def _result(self, is_healthy: bool, message: str, 
               start_time: float, details: Dict) -> HealthCheckResult:
        """Create a health check result"""
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            is_healthy=is_healthy,
            message=message,
            duration_ms=duration_ms,
            details=details
        )


class UptimeKumaClient:
    """Uptime Kuma integration"""
    
    def push(self, url: str, result: HealthCheckResult) -> bool:
        """Push status to Uptime Kuma"""
        try:
            params = {
                'status': HealthStatus.UP.value if result.is_healthy else HealthStatus.DOWN.value,
                'msg': result.message[:2000],  # Uptime Kuma has a limit
                'ping': int(result.duration_ms)
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            logging.info(f"Pushed to Uptime Kuma: {params['status']}")
            return True
            
        except requests.RequestException as e:
            logging.error(f"Failed to push to Uptime Kuma: {e}")
            return False


class OutputFormatter:
    """Formats output for different targets"""
    
    @staticmethod
    def format(result: HealthCheckResult, config: DiskmonConfig) -> str:
        """Format output based on configuration"""
        if config.output_format == OutputFormat.KUMA_ONLY:
            return ""  # No output
        
        if config.output_format == OutputFormat.JSON:
            return OutputFormatter._format_json(result, config)
        
        # Human format
        return OutputFormatter._format_human(result, config)
    
    @staticmethod
    def _format_json(result: HealthCheckResult, config: DiskmonConfig) -> str:
        """Format as JSON"""
        output = {
            'device': config.device,
            'status': 'UP' if result.is_healthy else 'DOWN',
            'healthy': result.is_healthy,
            'message': result.message,
            'duration_ms': round(result.duration_ms, 1),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'details': result.details
        }
        return json.dumps(output, indent=2)
    
    @staticmethod
    def _format_human(result: HealthCheckResult, config: DiskmonConfig) -> str:
        """Format for human reading with optional color"""
        status = 'UP' if result.is_healthy else 'DOWN'
        
        if config.color_output and sys.stdout.isatty():
            # Use color codes
            if result.is_healthy:
                color = '\033[92m'  # Green
                symbol = '✓'
            else:
                color = '\033[91m'  # Red
                symbol = '✗'
            reset = '\033[0m'
            
            output = f"{color}[{status}]{reset} {symbol} {result.message}\n"
            output += f"Device: {config.device}\n"
            output += f"Duration: {result.duration_ms:.1f}ms\n"
            
            # Add details if available
            if result.details:
                output += "\nDetails:\n"
                for key, value in result.details.items():
                    if key not in ['idle', 'idle_reason', 'smart_skipped']:
                        output += f"  • {key}: {value}\n"
        else:
            # No color
            output = f"[{status}] {result.message}\n"
            output += f"Device: {config.device}\n"
            output += f"Duration: {result.duration_ms:.1f}ms\n"
            
            if result.details:
                output += "\nDetails:\n"
                for key, value in result.details.items():
                    if key not in ['idle', 'idle_reason', 'smart_skipped']:
                        output += f"  - {key}: {value}\n"
        
        return output


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: DiskmonConfig):
    """Configure logging based on configuration"""
    log_format = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    log_level = logging.DEBUG if config.verbose else logging.INFO
    
    # Create log directory
    log_path = Path(config.log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Cannot create log directory: {e}", file=sys.stderr)
        sys.exit(1)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    handlers = [file_handler]
    
    # Console handler for verbose mode
    if config.verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )


# ============================================================================
# Signal Handling
# ============================================================================

class SignalHandler:
    """Handle graceful shutdown"""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown_requested = True
        logging.info(f"Received signal {signum}, shutting down gracefully")


# ============================================================================
# Main Execution
# ============================================================================

def main() -> int:
    """Main entry point"""
    # Start timing immediately
    start_time = time.time()
    
    # Check privileges
    PrivilegeChecker.check()
    
    # Parse arguments
    parser = ConfigManager.get_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args)
    config = config_manager.load()
    
    # Setup logging
    setup_logging(config)
    logging.info(f"Starting diskmon for {config.device}")
    logging.debug(f"Configuration: {config}")
    
    # Setup signal handling
    signal_handler = SignalHandler()
    
    # Initialize state manager
    state_manager = StateManager(config.device)
    
    # Main execution with lock
    try:
        with state_manager.lock:
            # Check if shutdown was requested during lock wait
            if signal_handler.shutdown_requested:
                logging.info("Shutdown requested before check could start")
                return 0
            
            # Load state
            state = state_manager.load()
            
            # Perform health check
            health_checker = HealthChecker(state_manager)
            result = health_checker.check(config, state)
            
            # Update state
            state.last_check_ts = time.time()
            state_manager.save(state)
            
            # Push to Uptime Kuma
            kuma_client = UptimeKumaClient()
            kuma_success = kuma_client.push(config.kuma_url, result)
            
            if not kuma_success:
                logging.warning("Failed to push status to Uptime Kuma")
            
            # Log result
            log_level = logging.INFO if result.is_healthy else logging.WARNING
            logging.log(
                log_level,
                f"Check complete: {'UP' if result.is_healthy else 'DOWN'} - "
                f"{result.message} ({result.duration_ms:.1f}ms)"
            )
            
            # Output result
            output = OutputFormatter.format(result, config)
            if output:
                print(output.rstrip())
            
            # Return appropriate exit code
            return 0 if result.is_healthy else 1
    
    except filelock.Timeout:
        logging.error(f"Could not acquire lock for {config.device} - another instance may be running")
        print(f"ERROR: Could not acquire lock for {config.device}", file=sys.stderr)
        return 2
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
