# Discrete PID Reminder

## Equations

```text
error = setpoint - measured_value
integral_state += error * dt
derivative = (error - previous_error) / dt
output = kp * error + ki * integral_state + kd * derivative
```

Use `derivative = 0` on the first update. Reject `dt <= 0`.

## API

```text
PIDController(kp, ki, kd)
update(setpoint, measured_value, dt) -> output
reset() -> None
```

State:

```text
kp, ki, kd
integral_state
previous_error
```

## Usage

Create the controller once, before the control loop:

```python
controller = PIDController(kp=2.0, ki=0.1, kd=0.5)

while running:
    measurement = read_sensor()
    output = controller.update(setpoint, measurement, dt)
    send_to_actuator(output)
```

Call `reset()` when starting a new control task. The output may represent
torque, voltage, force, or another actuator command.

## Checklist

- [ ] Implement `__init__`, `update()`, and `reset()`.
- [ ] Return the combined P, I, and D output.
- [ ] Preserve state between updates.
- [ ] Test zero error, integral behavior, derivative behavior, invalid `dt`,
      and reset.
- [ ] Add output limits and anti-windup later.
