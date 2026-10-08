# SpatialVector-HMI — Arduino Haptic Controller Protocol Specification

> **Single Source of Truth** for wire communication between Laptop (M10 Python Interface) and Arduino Firmware (`haptic_controller.ino`).
> Any modification to commands, pattern IDs, or timings MUST be updated here first.

---

## 1. Physical Interface & Serial Settings

- **Baud Rate**: `115200`
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Line Ending**: Newline (`\n` or `\r\n`)
- **Motor Pin Mapping**:
  - **Left Motor**: Digital/PWM Pin `5`
  - **Center Motor**: Digital/PWM Pin `6`
  - **Right Motor**: Digital/PWM Pin `9`

---

## 2. Command Format (Laptop -> Arduino)

All commands are single-line ASCII strings terminated by `\n`.

### 2.1 Haptic Actuation Command (`CMD`)
```text
CMD,<direction>,<urgency>,<pattern_id>,<duration_ms>\n
```

- `<direction>`: Motor target. One of:
  - `LEFT` — Left motor active
  - `CENTER` — Center motor active
  - `RIGHT` — Right motor active
  - `STOP` — Dual/All motors active or system stop pattern
- `<urgency>`: Integer `1` to `5`
  - `1`: Low / Clear
  - `2`: Mild / Caution
  - `3`: Moderate
  - `4`: High / Fast alert
  - `5`: Critical / Emergency
- `<pattern_id>`: Exact pattern string matching M09 `CorridorPolicy`:
  - `ALL_CLEAR`: Urgency 1, single subtle 200ms pulse (direction STOP)
  - `DEGRADED_WARN`: Urgency 2, double short pulse 350ms (direction STOP)
  - `LEFT_SLOW`, `LEFT_MED`, `LEFT_FAST`
  - `CENTER_SLOW`, `CENTER_MED`, `CENTER_FAST`
  - `RIGHT_SLOW`, `RIGHT_MED`, `RIGHT_FAST`
  - `STOP_SLOW`, `STOP_CRITICAL`
- `<duration_ms>`: Total execution duration in milliseconds (e.g. 200, 300, 350, 400, 500).

**Examples:**
```text
CMD,STOP,1,ALL_CLEAR,200
CMD,STOP,2,DEGRADED_WARN,350
CMD,LEFT,4,LEFT_FAST,200
CMD,RIGHT,3,RIGHT_MED,300
CMD,STOP,5,STOP_CRITICAL,500
```

### 2.2 System & Diagnostic Commands
- `TEST\n`: Initiates self-test pulse sequence across Left -> Center -> Right motors.
- `PING\n`: Heartbeat check from host.
- `STOP_ALL\n`: Emergency command to immediately stop all motor vibration.

---

## 3. Response Format (Arduino -> Laptop)

All responses are single-line ASCII strings terminated by `\n`.

### 3.1 Command Acknowledgment (`ACK`)
Emitted immediately upon receiving a valid `CMD`:
```text
ACK,<pattern_id>\n
```
*Example:* `ACK,LEFT_FAST\n`

### 3.2 Heartbeat Response (`PONG`)
```text
PONG\n
```

### 3.3 Self-Test Result (`TEST_RES`)
```text
TEST_RES,L:PASS,C:PASS,R:PASS\n
```

### 3.4 Error (`ERR`)
```text
ERR,<reason>\n
```

---

## 4. Hardware Watchdog Discipline (Non-Negotiable)

1. **Autonomous Timeout**: If the Arduino firmware does not receive any valid `CMD` within **500ms** (`WATCHDOG_TIMEOUT_MS`), it MUST immediately turn OFF all motors (`analogWrite(pin, 0)`).
2. **Fail-Safe**: The laptop should never be assumed to stay connected or alive. If the laptop freezes, crashes, or the USB cable disconnects, the motors must never be left buzzing indefinitely.
3. **Non-Blocking Safety Loop**: The laptop safety loop transmits `CMD` without waiting for an `ACK`. ACKs are consumed asynchronously to update `ArduinoStatus.last_ack_t` and connection status.
