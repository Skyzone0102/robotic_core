class DCMotor:
    """Simple brushed DC motor model."""

    def __init__(
        self,
        resistance=2.0,
        inductance=0.5,
        torque_constant=0.1,
        back_emf_constant=0.1,
        inertia=0.01,
        damping=0.001,
        max_voltage=12.0,
    ):
        self.resistance = resistance
        self.inductance = inductance
        self.torque_constant = torque_constant
        self.back_emf_constant = back_emf_constant
        self.inertia = inertia
        self.damping = damping
        self.max_voltage = max_voltage

        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0

    def update(self, voltage, dt, load_torque=0.0):
        voltage = max(-self.max_voltage, min(voltage, self.max_voltage))

        # Electrical dynamics:
        # V = R*i + L*di/dt + back_emf
        back_emf = self.back_emf_constant * self.velocity
        current_derivative = (
            voltage
            - self.resistance * self.current
            - back_emf
        ) / self.inductance

        self.current += current_derivative * dt

        # Mechanical dynamics:
        # J*domega/dt = motor_torque - damping - load_torque
        motor_torque = self.torque_constant * self.current
        acceleration = (
            motor_torque
            - self.damping * self.velocity
            - load_torque
        ) / self.inertia

        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position