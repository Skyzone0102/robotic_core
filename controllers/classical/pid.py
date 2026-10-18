class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = None

    def update(self, setpoint, measured_value, dt):
        # Validate dt
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = setpoint - measured_value
        # proportional term
        proportional = self.kp * error
        # integral term
        self.integral += error * dt
        # derivative term
        if self.previous_error is None:
            derivative = 0
        else:
            derivative = (error - self.previous_error) / dt

        # PID output
        output = proportional + (self.ki * self.integral) + (self.kd * derivative)

        # Update previous error for next iteration
        self.previous_error = error
        
        return output

    def reset(self):
        self.integral = 0
        self.previous_error = None