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
    
    # Idle detection
    idle_threshold: int = 300  # seconds
    max_idle_skip: int = 86400  # 24 hours
    
    # Behavior
    require_device_present: bool = True
    log_file: str = "/var/log/diskmon.log"
    output_format: OutputFormat = OutputFormat.HUMAN
    verbose: bool = False
    color_output: bool = True  # New: color support for human format


@dataclass
class DeviceState:
    """Persistent state for a monitored device"""
    device_path: str
    last_check_ts: Optional[float] = None
    last_short_test_scheduled_ts: Optional[float] = None
    last_long_test_scheduled_ts: Optional[float] = None
    last_badblocks_scheduled_ts: Optional[float] = None
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
    smart_status_passed: bool = True
    temperature: Optional[int] = None
    power_on_hours: Optional[int] = None
    attributes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    self_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_attribute_raw_value(self, attr_id: int) -> int:
        """Get raw value of a SMART attribute"""
        attr = self.attributes.get(attr_id, {})
        return attr.get('raw', {}).get('value', 0)
    
    def get_latest_self_test(self) -> Optional[Dict[str, Any]]:
        """Get most recent self-test result"""
        return self.self_tests[0] if self.self_tests else None


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    is_healthy: bool
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


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
        self._load_from_file()
        self._apply_cli_overrides()
        self._validate_config()
        return self.config
    
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
        """Parse SMART thresholds from CLI argument"""
        try:
            for item in self.args.smart.split(','):
                key, value = item.strip().split('=')
                key = key.strip()
                if key in self.config.smart_thresholds:
                    self.config.smart_thresholds[key] = int(value.strip())
                else:
                    logging.warning(f"Unknown SMART threshold key: {key}")
        except ValueError as e:
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
    
    @staticmethod
    def get_cli_parser() -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Diskmon - Linux Disk Health Monitoring Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Required arguments
        parser.add_argument('device', 
                          help='Block device path (e.g., /dev/sda)')
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
                          help='SMART thresholds: "pending=N,realloc=N,..."')
        
        # Test scheduling
        parser.add_argument('--short-test-days', type=int,
                          help='Days between short SMART tests (default: 7)')
        parser.add_argument('--long-test-days', type=int,
                          help='Days between long SMART tests (default: 30)')
        parser.add_argument('--badblocks-days', type=int,
                          help='Days between bad block scans (optional)')
        
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
        self.device_name = Path(device_path).name
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
            # Run a lightweight check first
            lightweight_check = self._run_smartctl(['-n', 'standby', '-j', '-H', device_path], timeout=30)
            
            if lightweight_check.returncode == 2:
                return (SmartStatus.SLEEPING, None)

            # If the command failed for any reason other than sleeping, try to parse its output anyway.
            # This allows us to handle the "checksum" and "test in progress" cases.
            if not lightweight_check.stdout:
                # If there's no output at all, it's a hard failure.
                logging.error(f"smartctl lightweight check failed with no output. stderr: {lightweight_check.stderr}")
                return (SmartStatus.FAILED, None)
            
            try:
                lightweight_data = json.loads(lightweight_check.stdout)
                
                # Check if a self-test is in progress
                if "in progress" in lightweight_data.get("self_test_status", {}).get("string", "").lower():
                    logging.info(f"SMART self-test is in progress. Performing a best-effort check.")
                    # Parse what we can from the lightweight data and return it for a partial check.
                    parsed_data = self._parse_json(lightweight_check.stdout)
                    return (SmartStatus.TEST_IN_PROGRESS, parsed_data)

            except json.JSONDecodeError:
                logging.warning("Could not parse lightweight smartctl JSON.")
                # Fall through to attempt the full check

            # If we are here, the disk is not sleeping and no test is in progress. Proceed with full check.
            result = self._run_smartctl(['-j', '-a', device_path], timeout=60)
            is_command_error = (result.returncode & 7) != 0

            if is_command_error:
                logging.error(f"smartctl command failed with critical error code {result.returncode}: {result.stderr}")
                return (SmartStatus.FAILED, None)
            
            if "invalid SMART checksum" in result.stderr:
                logging.warning("Proceeding with SMART data parsing despite checksum warning.")

            parsed_data = self._parse_json(result.stdout)
            if parsed_data is None: return (SmartStatus.FAILED, None)
            
            return (SmartStatus.SUCCESS, parsed_data)
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.error(f"SMART data collection failed: {e}")
            return (SmartStatus.FAILED, None)
    
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
        """Parse smartctl JSON output"""
        try:
            data = json.loads(json_str)
            smart_data = SMARTData()
            
            # Overall status
            smart_data.smart_status_passed = data.get('smart_status', {}).get('passed', False)
            
            # Temperature
            smart_data.temperature = data.get('temperature', {}).get('current')
            
            # Power on hours
            smart_data.power_on_hours = data.get('power_on_time', {}).get('hours')
            
            # Attributes
            for attr in data.get('ata_smart_attributes', {}).get('table', []):
                if 'id' in attr:
                    smart_data.attributes[attr['id']] = attr
            
            # Self-test log
            log_data = data.get('ata_smart_self_test_log', {})
            smart_data.self_tests = log_data.get('standard', {}).get('table', [])
            
            return smart_data
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse SMART JSON: {e}")
            return None
    
    def evaluate(self, data: SMARTData, config: DiskmonConfig, is_test_in_progress: bool = False) -> Tuple[bool, List[str]]:
        """Evaluate SMART data against thresholds"""
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


class HealthChecker:
    """Coordinates health checks"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.device_manager = DeviceManager()
        self.fs_monitor = FilesystemMonitor()
        self.smart_handler = SMARTHandler()
        
    def check(self, config: DiskmonConfig, state: DeviceState) -> HealthCheckResult:
        """Perform complete health check with fully robust 'test in progress' logic."""
        start_time = time.time()
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
                self._schedule_tests(config, state)

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
    
    def _schedule_tests(self, config: DiskmonConfig, state: DeviceState):
        """Schedule tests if due"""
        now = time.time()
        
        # Short test
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
