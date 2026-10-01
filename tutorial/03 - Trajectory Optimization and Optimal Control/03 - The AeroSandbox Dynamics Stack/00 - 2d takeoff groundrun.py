from aerosandbox.weights.mass_properties_of_shapes import mass_properties_from_radius_of_gyration
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

from cessna152 import airplane  # See cessna152.py for details.
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p



mass_props = mass_properties_from_radius_of_gyration(
    mass=1151.8 * u.lbm,
    radius_of_gyration_x=2,
    radius_of_gyration_y=3,
    radius_of_gyration_z=3,
)
mass_props.x_cg = 1


# Note: AeroBuildup is fully vectorized, so we evaluate all 500 alpha points simultaneously.




### Initialize the problem
opti = asb.Opti()

### Define time. Note that the horizon length is unknown.
time_final_guess = 100
time = np.cosspace(
    0,
    opti.variable(init_guess=time_final_guess, log_transform=True),
    100
)
N = np.length(time)

time_guess = np.linspace(0, time_final_guess, N)

### Create a dynamics instance
init_state = {
    "x_e"  : 0,
    "z_e"  : 0,  # 1 km altitude
    "speed": 10 * u.knot,
    "gamma": 0,
}

dyn = asb.DynamicsPointMass2DSpeedGamma(
    mass_props=mass_props,
    x_e=opti.variable(init_state["speed"] * time_guess),
    z_e=opti.variable(np.linspace(init_state["z_e"], 100, N)),
    speed=opti.variable(init_guess=init_state["speed"], n_vars=N),
    gamma=opti.variable(init_guess=0, n_vars=N, lower_bound=-np.pi / 2, upper_bound=np.pi / 2),
    alpha=opti.variable(init_guess=5, n_vars=N, lower_bound=-5, upper_bound=15),
    
)
# Constrain the initial state

print(dyn.state.keys())
#import pdb
#pdb.set_trace()

for k in dyn.state.keys():
    opti.subject_to(
        dyn.state[k][0] == init_state[k]
    )


### Add in forces
dyn.add_gravity_force(g=9.81)

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point
).run()

dyn.add_force(
    *aero["F_w"],
    axes="wind"
)

thrust = opti.variable(init_guess=3000, n_vars=N, lower_bound=0, upper_bound=5000)

#thrust = 3000
dyn.add_force(Fx =thrust,axes = "wind")

dyn.add_force(Fz=np.maximum(dyn.mass_props.mass * 9.81 - aero["L"],0) , axes="wind")


### Constrain the altitude to be above ground at all times
opti.subject_to(
    dyn.altitude > 0
)



### Finalize the problem
dyn.constrain_derivatives(opti, time)  # Apply the dynamics constraints created up to this point

#opti.minimize(-dyn.x_e[-1])  # Go as far downrange as you can
opti.minimize(-dyn.speed[-1])

### Solve it
sol = opti.solve()

opti.debug.value(dyn.x_e)

### Substitute the optimization variables in the dynamics instance with their solved values (in-place)
dyn = sol(dyn)

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
plt.sca(ax[0, 0])
plt.plot(dyn.x_e, dyn.altitude)
plt.xlabel("Range [m]")
plt.ylabel("Altitude [m]")

plt.sca(ax[0, 1])
plt.plot(sol(time), dyn.speed)
plt.xlabel("Time [sec]")
plt.ylabel("Speed (True) [m/s]")

plt.sca(ax[1, 0])
plt.plot(sol(time), dyn.alpha)
plt.xlabel("Time [sec]")
plt.ylabel("Angle of Attack [deg]")

plt.sca(ax[1, 1])
plt.plot(sol(time), np.degrees(dyn.gamma))
plt.xlabel("Time [sec]")
plt.ylabel("Flight Path Angle [deg]")

p.show_plot("Takeoff Ground Run")

# Add this after your existing plotting code

# Create 6 subplots for the 6 force components
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Wind Axes Forces
axes[0].plot(sol(time), dyn.Fx_w)
axes[0].set_title('Wind Axes - Fx (Drag)')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Force [N]')
axes[0].grid(True)

axes[1].plot(sol(time), dyn.Fz_w)
axes[1].set_title('Wind Axes - Fz (Lift)')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Force [N]')
axes[1].grid(True)

"""
axes[2].plot(sol(time), dyn.)
axes[2].set_title('Thrust')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Force [N]')
axes[2].grid(True)
"""

plt.tight_layout()
plt.show()