import matplotlib.pyplot as plt

from controllers.classical.pid import PIDController
from models.dc_motor import DCMotor

# tracks position over time
def main():
    controller = PIDController(kp=1.8, ki=0.05, kd=1.0)
    motor = DCMotor()

    setpoint = 1.0
    dt = 0.01
    simulation_time = 5.0
    steps = int(simulation_time / dt)

    times = []
    positions = []
    velocities = []
    commands = [] # motor input is voltage command

    for step in range(steps):
        time = step * dt

        command = controller.update(
            setpoint=setpoint,
            measured_value=motor.position,
            dt=dt,
        )

        position = motor.update(command, dt)

        times.append(time)
        positions.append(position)
        velocities.append(motor.velocity)
        commands.append(command)

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    axes[0].plot(times, positions, label="Motor position")
    axes[0].axhline(setpoint, color="red", linestyle="--", label="Setpoint")
    axes[0].set_ylabel("Position")
    axes[0].set_title("PID Motor Simulation")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, commands, label="Motor command", color="green")
    axes[1].plot(times, velocities, label="Motor velocity", color="orange")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Final position: {motor.position:.4f}")
    print(f"Target position: {setpoint:.4f}")

if __name__ == "__main__":
    main()