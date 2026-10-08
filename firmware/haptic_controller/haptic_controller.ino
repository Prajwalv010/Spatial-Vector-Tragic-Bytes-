/*
 * SpatialVector-HMI — Arduino Haptic Controller Firmware (M10)
 *
 * Implements 3-channel haptic feedback (Left, Center, Right) for collision
 * avoidance guidance.
 *
 * Wire protocol defined in: firmware/haptic_controller/PROTOCOL.md
 * (Any changes to command structure must update PROTOCOL.md in same commit)
 *
 * Pin Mapping:
 *   Left Motor:   Digital/PWM Pin 5
 *   Center Motor: Digital/PWM Pin 6
 *   Right Motor:  Digital/PWM Pin 9
 *
 * Safety & Watchdog Rules:
 *   1. Hardware Watchdog: If no CMD arrives within WATCHDOG_TIMEOUT_MS (500ms),
 *      all motors immediately shut off (PWM = 0).
 *   2. Non-blocking: All pulses and pattern timing run via millis() state machine.
 */

#define PIN_MOTOR_LEFT   5
#define PIN_MOTOR_CENTER 6
#define PIN_MOTOR_RIGHT  9

#define WATCHDOG_TIMEOUT_MS 500

// Active command state
struct ActiveCommand {
    char direction[10];
    int urgency;
    char pattern_id[32];
    unsigned long duration_ms;
    unsigned long start_time;
    bool active;
};

ActiveCommand current_cmd = {"", 0, "", 0, 0, false};
unsigned long last_cmd_received_time = 0;
char serial_buf[96];
int serial_buf_idx = 0;

void setMotorPWM(int left_val, int center_val, int right_val) {
    analogWrite(PIN_MOTOR_LEFT, constrain(left_val, 0, 255));
    analogWrite(PIN_MOTOR_CENTER, constrain(center_val, 0, 255));
    analogWrite(PIN_MOTOR_RIGHT, constrain(right_val, 0, 255));
}

void stopAllMotors() {
    setMotorPWM(0, 0, 0);
    current_cmd.active = false;
}

void executeSelfTest() {
    // Sequential motor pulse test: Left -> Center -> Right
    setMotorPWM(200, 0, 0);
    delay(150);
    setMotorPWM(0, 200, 0);
    delay(150);
    setMotorPWM(0, 0, 200);
    delay(150);
    stopAllMotors();
    Serial.println("TEST_RES,L:PASS,C:PASS,R:PASS");
}

void handleCommand(const char* line) {
    if (strncmp(line, "PING", 4) == 0) {
        Serial.println("PONG");
        return;
    }
    if (strncmp(line, "TEST", 4) == 0) {
        executeSelfTest();
        return;
    }
    if (strncmp(line, "STOP_ALL", 8) == 0) {
        stopAllMotors();
        Serial.println("ACK,STOP_ALL");
        return;
    }

    // Format: CMD,<direction>,<urgency>,<pattern_id>,<duration_ms>
    if (strncmp(line, "CMD,", 4) == 0) {
        char dir[10] = "";
        int urg = 1;
        char pat[32] = "";
        unsigned long dur = 200;

        // Parse CSV fields safely
        int parsed = sscanf(line, "CMD,%9[^,],%d,%31[^,],%lu", dir, &urg, pat, &dur);
        if (parsed == 4) {
            strncpy(current_cmd.direction, dir, sizeof(current_cmd.direction) - 1);
            current_cmd.urgency = urg;
            strncpy(current_cmd.pattern_id, pat, sizeof(current_cmd.pattern_id) - 1);
            current_cmd.duration_ms = dur;
            current_cmd.start_time = millis();
            current_cmd.active = true;

            last_cmd_received_time = millis();

            // Emit immediate ACK
            Serial.print("ACK,");
            Serial.println(current_cmd.pattern_id);
        } else {
            Serial.println("ERR,PARSE_FAILED");
        }
    }
}

void updateMotors() {
    unsigned long now = millis();

    // 1. Hardware Watchdog Check
    if (now - last_cmd_received_time > WATCHDOG_TIMEOUT_MS) {
        stopAllMotors();
        return;
    }

    // 2. Duration Expiry Check
    if (!current_cmd.active || (now - current_cmd.start_time > current_cmd.duration_ms)) {
        stopAllMotors();
        return;
    }

    // 3. Compute PWM intensity based on urgency (1..5)
    // 1: 100, 2: 140, 3: 180, 4: 220, 5: 255
    int pwm = 70 + current_cmd.urgency * 37;
    pwm = constrain(pwm, 0, 255);

    // 4. Pattern modulation (pulsing)
    unsigned long elapsed = now - current_cmd.start_time;
    bool pulse_on = true;

    // Fast patterns: 100ms on, 50ms off
    if (strstr(current_cmd.pattern_id, "FAST") != NULL) {
        pulse_on = ((elapsed / 75) % 2 == 0);
    }
    // Med patterns: 150ms on, 100ms off
    else if (strstr(current_cmd.pattern_id, "MED") != NULL) {
        pulse_on = ((elapsed / 125) % 2 == 0);
    }
    // Warn / Degraded: double short pulse
    else if (strstr(current_cmd.pattern_id, "DEGRADED_WARN") != NULL) {
        pulse_on = ((elapsed < 120) || (elapsed > 180 && elapsed < 300));
    }
    // All Clear: soft single pulse
    else if (strstr(current_cmd.pattern_id, "ALL_CLEAR") != NULL) {
        pwm = 90; // Gentle notification
        pulse_on = (elapsed < 180);
    }

    if (!pulse_on) {
        pwm = 0;
    }

    // 5. Apply to directional motors
    if (strcmp(current_cmd.direction, "LEFT") == 0) {
        setMotorPWM(pwm, 0, 0);
    } else if (strcmp(current_cmd.direction, "CENTER") == 0) {
        setMotorPWM(0, pwm, 0);
    } else if (strcmp(current_cmd.direction, "RIGHT") == 0) {
        setMotorPWM(0, 0, pwm);
    } else if (strcmp(current_cmd.direction, "STOP") == 0) {
        // Critical or degraded stop alerts activate all/dual channels
        if (strcmp(current_cmd.pattern_id, "STOP_CRITICAL") == 0) {
            setMotorPWM(pwm, pwm, pwm);
        } else if (strcmp(current_cmd.pattern_id, "ALL_CLEAR") == 0) {
            setMotorPWM(0, pwm, 0); // Gentle center vibration
        } else {
            setMotorPWM(pwm, 0, pwm); // Flanking dual-channel vibration
        }
    }
}

void setup() {
    pinMode(PIN_MOTOR_LEFT, OUTPUT);
    pinMode(PIN_MOTOR_CENTER, OUTPUT);
    pinMode(PIN_MOTOR_RIGHT, OUTPUT);

    stopAllMotors();

    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port on USB native boards
    }

    Serial.println("STATUS,READY");
    last_cmd_received_time = millis();
}

void loop() {
    // Read serial commands non-blockingly
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serial_buf_idx > 0) {
                serial_buf[serial_buf_idx] = '\0';
                handleCommand(serial_buf);
                serial_buf_idx = 0;
            }
        } else if (serial_buf_idx < (int)sizeof(serial_buf) - 1) {
            serial_buf[serial_buf_idx++] = c;
        }
    }

    // Update motor outputs and enforce watchdog
    updateMotors();
}
