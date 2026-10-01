import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import pdb 
from aerosandbox.common import AeroSandboxObject
from aerosandbox.atmosphere._isa_atmo_functions import pressure_isa, temperature_isa
from aerosandbox.atmosphere import Atmosphere
import aerosandbox.tools.units as u
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.library import propulsion_turbofan as lib_prop_turbofan

opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='all'
)


g_m_s2 = 9.81
make_plots = True

assets = Path("assets")

mtow = opti.variable(init_guess=10, log_transform=True, lower_bound=100, upper_bound=1000)  # takeoff weight, N
wing_loading = opti.variable(init_guess=10, log_transform=True, lower_bound=100, upper_bound=1000)  # wing loading, N/m^2
thrust_to_weight = opti.variable(init_guess=10, log_transform=True, lower_bound=0.05, upper_bound=0.5)  # thrust to weight ratio


aspect_ratio = opti.variable(init_guess=5.83845, log_transform=True)  # aspect ratio
S = mtow / wing_loading
B = np.sqrt(aspect_ratio * S)



# TODO: Parametric root chord and MAC values

sect1_tpr = 0.59375
sect2_tpr = 0.3159
sect3_tpr = 0.4
sect4_tpr = 0.33

taper_ratio = sect1_tpr * sect2_tpr * sect3_tpr * sect4_tpr
c_root = S *2 / B / (1+ taper_ratio) *2

airplane = asb.Airplane(
    name="NACA_RM_A50K27 Wing",
    #xyz_ref=[21.99, 0, 0.116],  # CG location
    xyz_ref=[c_root/2, 0, 0.116],
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=[0.0, 0, 0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=1 * c_root,
                    twist=0.0,  # degrees
                    airfoil=asb.Airfoil(
                        name="n64_1_A612",
                        coordinates="tutorial/02 - Design Optimization/n64_1_A612.dat"
                    ),  # Airfoils are blended between a given XSec and the next one.
                ),
                asb.WingXSec(
                    xyz_le=[12.25/40.14 * c_root, B * 5 / 26, 0], # x = 12.25, y = 6.25 
                    chord=23.75 / 40.14 * c_root,
                    twist=0,
                    airfoil=asb.Airfoil(
                        name="n64_1_A612",
                        coordinates="tutorial/02 - Design Optimization/n64_1_A612.dat"
                    ),
                ),
                asb.WingXSec(
                    xyz_le=[24.5/40.14 * c_root, B * 5/13, 0], # x = 24.5,y = 12.5
                    chord=7.5 / 40.14 * c_root,
                    twist=0,
                    airfoil=asb.Airfoil(
                        name="n64_1_A612",
                        coordinates="tutorial/02 - Design Optimization/n64_1_A612.dat"
                    ),
                ),
                asb.WingXSec(
                    xyz_le=[9/10 * c_root, 32.5 /35.5 * B/2, 0], #x = 36 y = 32.5
                    chord=3 / 40.14 * c_root,
                    twist=0,
                    airfoil=asb.Airfoil(
                        name="n64_1_A612",
                        coordinates="tutorial/02 - Design Optimization/n64_1_A612.dat"
                    ),
                ),
            ]
        )
    ],
    #c_ref=13.687 
    c_ref = 2/3 * c_root * (1 + taper_ratio + taper_ratio ** 2)/ (1 + taper_ratio + taper_ratio ** 2)

)

S_calc = airplane.wings[0].area()

xyz_ref = [c_root/2, 0, 0.116]

m_dot_core_corrected
mass_turbofan = lib_prop_turbofan.mass_turbofan(
    m_dot_core_corrected,
    overall_pressure_ratio = 12,
    bypass_ratio = 9,
    diameter_fan = 3,
)
thrust = lib_prop_turbofan.thrust_turbofan(mass_turbofan = mass_turbofan)

cruise_power_propulsion = lib_prop_turbofan.propeller_shaft_power_from_thrust(
    thrust_force=design_thrust_cruise_total,
    area_propulsive=propulsive_area_total,
    airspeed=cruise_op_point.velocity,
    rho=cruise_op_point.atmosphere.density(),
    propeller_coefficient_of_performance=propeller_coefficient_of_performance
) / motor_efficiency


##### Section: Aerodynamics
cruise_alt = 30000 * u.foot
climb_alt = 1000 * u.foot
cruise_op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=14,
        lower_bound=4,
        upper_bound=150,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=10,
    ),
    atmosphere = asb.Atmosphere(altitude=cruise_alt, temperature_deviation=0),
)

cruise_aeros = asb.AeroBuildup(
    airplane=airplane,
    op_point=cruise_op_point,
    xyz_ref=xyz_ref,
).run()

climb_op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=14,
        lower_bound=4,
        upper_bound=150,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=10,
    ),
    atmosphere = asb.Atmosphere(altitude=climb_alt, temperature_deviation=0),
)


climb_aeros = asb.AeroBuildup(
    airplane=airplane,
    op_point=climb_op_point,
    xyz_ref=xyz_ref,
).run()


cruise_ld = cruise_aeros["L"] / cruise_aeros["D"]
climb_ld = climb_aeros["L"] / climb_aeros["D"]

opti.subject_to([
    cruise_ld < 0.75 * climb_ld,
])

opti.minimize(-cruise_ld)

sol = opti.solve()

"""
vs1_m_s = mass_props_TOGW.weight / (0.5 * atmo.density() * airplane.wings[0].area() * CL_max)
vr_m_s = 1.1 * vs1_m_s
vr_srt_2_m_s = vr_m_s / np.sqrt(2)
vr_srt_2_m_s = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=vr_srt_2_m_s,
        lower_bound=4,
        upper_bound=100,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=10,
    )
)

gamma_climb_deg = 10 # degrees
gamma_climb_rad = np.deg2rad(gamma_climb_deg)

h_obst_ft = 35
h_obst_m = h_obst_ft * 0.3048

mu_rolling = 0.03 # ground friction coefficient

# GA Eq 18-22
s_gr_m = vr_m_s **2 * (mtow_kg) / ((0.5 * rho_kg_m3 * S_m2 * CL_max) * 2 * g_m_s2)

v_lo_m_s = 1.1 * v_s1_m_s

# GA Eq 18-26 
s_rot_m = 3 * np.abs(v_lo_m_s)

# GA Eq 18-27
r_trans_m = 0.2156 * v_s1_m_s **2

# GA Eq 18-29
h_tr_m = r_trans_m * (1 - np.cos(gamma_climb_rad))

# GA Eq 18-30 / 18 - 28 
s_tr_m = np.sqrt(r_trans_m**2 - (r_trans_m - h_tr_m)**2)

# GA Eq 18-32
s_obs_clb_m = (h_obst_m - h_tr_m) / np.tan(gamma_climb_rad)

if h_tr_m < h_obst_m:
    s_obs_m = s_obs_clb_m + s_tr_m
else:
    # GA Eq 18-31
    s_obs_m = np.sqrt(r_trans_m**2 - (r_trans_m - h_obst_m)**2)

"""
