import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the AeroSandbox development directory to the Python path
asb_root = Path(__file__).parent.parent
sys.path.insert(0, str(asb_root))

import aerosandbox as asb
import aerosandbox.numpy as np  # AeroSandbox's numpy
import aerosandbox.library.aerodynamics as lib_aero
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
import aerosandbox.tools.units as u
import copy

# BoTorch imports
try:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from botorch.acquisition import qLogNoisyExpectedImprovement, qExpectedImprovement
    from botorch.acquisition.objective import ConstrainedMCObjective
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    print("BoTorch not available. Install with: pip install botorch")
    BOTORCH_AVAILABLE = False


def fit_single_task_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None = None,
) -> SingleTaskGP:
    """Fit a SingleTaskGP model."""
    if not BOTORCH_AVAILABLE:
        return None
        
    # Instantiate a GP model
    model = SingleTaskGP(
        train_x,
        train_y,
        train_yvar,
        input_transform=Normalize(d=train_x.shape[-1]),
    )

    # Train the hyperparameters
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def aerosandbox_objective_with_constraints(x: torch.Tensor) -> torch.Tensor:
    """
    Run AeroSandbox simulation for given design parameters with constraints.
    x: torch.Tensor of shape (n, 2) with [span, mean_chord], normalized to [0, 1].
    Returns: [climb_rate, constraint1, constraint2, constraint3, constraint4, constraint5] 
             where constraints <= 0 when satisfied, shape (n, 6).
    """
    # Denormalize inputs to physical ranges
    span_bounds = torch.tensor([5.0, 15.0], device=x.device)
    chord_bounds = torch.tensor([0.5, 2.0], device=x.device)
    span = x[:, 0] * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
    mean_chord = x[:, 1] * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]

    # Convert to regular numpy for AeroSandbox
    span_np = span.detach().cpu().numpy()
    chord_np = mean_chord.detach().cpu().numpy()
    
    results = []
    
    # Process each design individually (AeroSandbox doesn't support batching)
    for i in range(x.shape[0]):
        # Create a complete airplane similar to design_opt.py
        # Basic configuration
        x_tail = 0 + span_np[i] * 0.38
        
        # Wing with proper airfoils and twist
        wing = asb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=chord_np[i],
                    twist=4.70,
                    airfoil=asb.Airfoil("ag34")
                ),
                asb.WingXSec(
                    xyz_le=[0, span_np[i] / 2 * 0.6, 0],  # 60% break point
                    chord=chord_np[i],
                    twist=3.50,
                    airfoil=asb.Airfoil("ag34")
                ),
                asb.WingXSec(
                    xyz_le=[0, span_np[i] / 2, 0],
                    chord=chord_np[i] * 0.8,  # Tapered tip
                    twist=3.00,
                    airfoil=asb.Airfoil("ag36")
                )
            ]
        )
        
        # VStab (vertical stabilizer)
        vstab_volume_coefficient = 0.030
        vstab_aspect_ratio = 2.5
        vstab_taper_ratio = 0.7
        
        vstab_area = vstab_volume_coefficient * wing.area() * wing.span() / (x_tail - chord_np[i] / 4)
        vstab_span = (vstab_area * vstab_aspect_ratio) ** 0.5
        vstab_chord = (vstab_area / vstab_aspect_ratio) ** 0.5
        
        vstab = asb.Wing(
            name="VStab",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[-vstab_chord * vstab_taper_ratio ** -0.5, 0, 0],
                    chord=vstab_chord * vstab_taper_ratio ** -0.5,
                    airfoil=asb.Airfoil("ht14"),
                ),
                asb.WingXSec(
                    xyz_le=[-vstab_chord * vstab_taper_ratio ** 0.5, 0, vstab_span],
                    chord=vstab_chord * vstab_taper_ratio ** 0.5,
                    airfoil=asb.Airfoil("ht14"),
                )
            ]
        ).translate([x_tail, 0, 0])
        
        # HStab (horizontal stabilizer)
        hstab_volume_coefficient = 0.47
        hstab_aspect_ratio = 4.5
        
        hstab_area = hstab_volume_coefficient * wing.area() * wing.mean_aerodynamic_chord() / (x_tail - vstab_chord)
        hstab_span = (hstab_area * hstab_aspect_ratio) ** 0.5
        hstab_chord = (hstab_area / hstab_aspect_ratio) ** 0.5
        
        hstab = asb.Wing(
            name="HStab",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, y, 0],
                    chord=hstab_chord,
                    twist=4.00,
                    airfoil=asb.Airfoil("ht14"),
                )
                for y in [0, hstab_span / 2]
            ]
        ).translate([vstab.xsecs[-1].xyz_le[0], 0, vstab_span])
        
        # Fuselage (simplified)
        fuselage_length = (10 / 4) ** (1 / 2)  # Fixed mass assumption
        x_fuse_nose = -0.35 * fuselage_length
        
        # Create complete airplane
        airplane = asb.Airplane(
            name="SolarSeaplane",
            wings=[wing, hstab, vstab]
        )
        
        # Run aerodynamics analysis
        cruise_op_point = asb.OperatingPoint(
            velocity=14.0,  # m/s (from design_opt.py cruise speed)
            alpha=0.0,      # degrees (from design_opt.py cruise alpha)
            beta=0.0
        )
            
        aero = asb.AeroBuildup(
            airplane=airplane,
                    op_point=cruise_op_point
        ).run()
        
        # Get L/D ratio
        LD_cruise = aero["CL"] / aero["CD"]
        
        # Use exact physics from design_opt.py (no thrust convergence)
        
        # Fixed design parameters (from design_opt.py)
        g = 9.81
        structural_mass_markup = 1.2  # over Enigma F5J
        
        # Initial mass estimate for iteration (will be updated)
        design_mass_TOGW = 10.0  # kg (initial guess)
        
        # Solar panel arrangement (exact from design_opt.py)
        panel_spacing = 0.127 - 1e-6  # center to center
        solar_area_per_panel = (0.125) ** 2
        wing_extra_span = 1.5 * 0.0254  # 1.5 inches in meters
        
        # Calculate actual wing span and solar area based on design
        wing_span_actual = span_np[i]
        wing_root_chord_actual = chord_np[i]
        wing_tip_chord_actual = chord_np[i] * 0.8  # Tapered tip
        
        # Solar area calculation (exact from design_opt.py)
        wing_y_break_fraction = 0.6  # Simplified from panel arrangement
        n_panels_spanwise_inboard = int(wing_span_actual * wing_y_break_fraction / panel_spacing)
        n_panels_spanwise_outboard = int(wing_span_actual * (1 - wing_y_break_fraction) / panel_spacing)
        n_panels_chordwise_inboard = int(wing_root_chord_actual * 0.9 / panel_spacing)
        n_panels_chordwise_outboard = int(wing_tip_chord_actual * 0.9 / panel_spacing)
        
        n_panels = 2 * (
            n_panels_spanwise_inboard * n_panels_chordwise_inboard +
            n_panels_spanwise_outboard * n_panels_chordwise_outboard
        )
        solar_area = n_panels * solar_area_per_panel
        
        # Mass calculations (exact from design_opt.py)
        mass_wing = (
            (0.440 + 0.460) *
            (wing.area() / (0.264 * 3.624 * np.pi / 4)) ** 0.758 *
            (design_mass_TOGW / 1.475) ** 0.49 *
            (wing.aspect_ratio() / 18) ** 0.6
        ) * structural_mass_markup
        
        mass_vstab = (
            0.055 *
            (
                0.3 * (design_mass_TOGW / 1.475) ** 0.40 * (vstab_span / (0.670 / np.cos(np.deg2rad(40)))) ** 1.58 +
                0.7 * (chord_np[i] / 0.264) * (vstab_span / (0.670 / np.cos(np.deg2rad(40))))
            )
        ) * structural_mass_markup
        
        mass_hstab = (
            0.055 *
            (
                0.4 * (design_mass_TOGW / 1.475) ** 0.40 * (hstab_span / (0.670 / np.cos(np.deg2rad(40)))) ** 1.58 +
                0.6 * (chord_np[i] / 0.264) * (hstab_span / (0.670 / np.cos(np.deg2rad(40))))
            )
        ) * structural_mass_markup
        
        # Fuselage and boom (exact from design_opt.py)
        fuselage_length = (design_mass_TOGW / 4) ** (1 / 2)
        mass_boom = (
            0.235 *
            (x_tail / 1.675) *
            (design_mass_TOGW / 1.475) ** 0.49
        ) * structural_mass_markup
        
        # Solar cells mass (exact from design_opt.py)
        rho_solar_cells = 0.425 * 1.1 * 1.15  # kg/m^2
        mass_solar_cells = solar_area * rho_solar_cells
        
        # Fixed masses (exact from design_opt.py)
        mass_avionics = 0.060  # RX, pixhawk mini
        mass_servos = 0.050
        mass_mppts = (n_panels / 36) * 0.080  # Genasun GV-5
        mass_wiring = (n_panels / 72) * 0.100
        
        # Thrust requirements (exact from design_opt.py)
        design_thrust_cruise_total = design_mass_TOGW * g / LD_cruise
        
        # Propulsion setup (exact from design_opt.py)
        n_propellers = 2
        ideal_propeller_pressure_jump = 0.20 * cruise_op_point.dynamic_pressure()
        
        propeller_diameter = np.minimum(
            (4 / np.pi * design_thrust_cruise_total / ideal_propeller_pressure_jump / n_propellers) ** 0.5,
            fuselage_length * (10 * 0.0254)  # 10 inches in meters
        )
        
        propulsive_area_total = n_propellers * (np.pi / 4) * propeller_diameter ** 2
        
        # Propeller performance (exact from design_opt.py)
        propeller_tip_mach = 0.36
        propeller_rads_per_sec = propeller_tip_mach * cruise_op_point.atmosphere.speed_of_sound() / (propeller_diameter / 2)
        propeller_rpm = propeller_rads_per_sec * 30 / np.pi
        
        motor_efficiency = 0.75
        propeller_coefficient_of_performance = 0.90
        
        # Propulsion power (exact from design_opt.py)
        cruise_power_propulsion = lib_prop_prop.propeller_shaft_power_from_thrust(
            thrust_force=design_thrust_cruise_total,
            area_propulsive=propulsive_area_total,
            airspeed=cruise_op_point.velocity,
            rho=cruise_op_point.atmosphere.density(),
            propeller_coefficient_of_performance=propeller_coefficient_of_performance
        ) / motor_efficiency
        
        # Motor and propeller masses (exact from design_opt.py)
        motor_kv = propeller_rpm / (4 * 3.7)  # battery voltage
        mass_motors = lib_prop_elec.mass_motor_electric(
            max_power=cruise_power_propulsion / n_propellers,
            kv_rpm_volt=motor_kv,
            voltage=4 * 3.7,
        ) * n_propellers
        
        mass_propellers = n_propellers * lib_prop_prop.mass_hpa_propeller(
            diameter=propeller_diameter,
            max_power=cruise_power_propulsion / n_propellers,
            include_variable_pitch_mechanism=False
        )
        
        mass_escs = lib_prop_elec.mass_ESC(max_power=cruise_power_propulsion / n_propellers) * n_propellers
        
        # Wing sponsons mass (exact from design_opt.py)
        wing_sponson_length = chord_np[i] * 0.75
        wing_sponson_diameter = wing_sponson_length * 0.3
        wing_sponson_wetted_area = wing_sponson_length * (np.pi * wing_sponson_diameter)
        mass_wing_sponsons = 2 * wing_sponson_wetted_area * (2 * (4 * 0.0283495 / 0.836127 ** 2)) * 1.5
        
        # Total mass (exact from design_opt.py)
        total_mass = (
            mass_wing + mass_vstab + mass_hstab + mass_boom + 
            mass_solar_cells + mass_avionics + mass_servos + 
            mass_mppts + mass_wiring + mass_motors + 
            mass_propellers + mass_escs + mass_wing_sponsons
        )
        
        # Add glue weight (exact from design_opt.py)
        total_mass += total_mass * 0.08
        
        # Mass convergence loop (exact from design_opt.py)
        mass_converged = False
        max_iterations = 10
        iteration = 0
        
        while not mass_converged and iteration < max_iterations:
            iteration += 1
            old_mass = design_mass_TOGW
            
            # Update mass-dependent calculations
            mass_wing = (
                (0.440 + 0.460) *
                (wing.area() / (0.264 * 3.624 * np.pi / 4)) ** 0.758 *
                (design_mass_TOGW / 1.475) ** 0.49 *
                (wing.aspect_ratio() / 18) ** 0.6
            ) * structural_mass_markup
            
            mass_vstab = (
                0.055 *
                (
                    0.3 * (design_mass_TOGW / 1.475) ** 0.40 * (vstab_span / (0.670 / np.cos(np.deg2rad(40)))) ** 1.58 +
                    0.7 * (chord_np[i] / 0.264) * (vstab_span / (0.670 / np.cos(np.deg2rad(40))))
                )
            ) * structural_mass_markup
            
            mass_hstab = (
                0.055 *
                (
                    0.4 * (design_mass_TOGW / 1.475) ** 0.40 * (hstab_span / (0.670 / np.cos(np.deg2rad(40)))) ** 1.58 +
                    0.6 * (chord_np[i] / 0.264) * (vstab_span / (0.670 / np.cos(np.deg2rad(40))))
                )
            ) * structural_mass_markup
            
            mass_boom = (
                0.235 *
                (x_tail / 1.675) *
                (design_mass_TOGW / 1.475) ** 0.49
            ) * structural_mass_markup
            
            # Recalculate total mass
            total_mass = (
                mass_wing + mass_vstab + mass_hstab + mass_boom + 
                mass_solar_cells + mass_avionics + mass_servos + 
                mass_mppts + mass_wiring + mass_motors + 
                mass_propellers + mass_escs + mass_wing_sponsons
            )
            total_mass += total_mass * 0.08
            
            # Update design mass for next iteration
            design_mass_TOGW = total_mass

            # Print Iteration
            print(f"Iteration {iteration}: design_mass_TOGW = {float(design_mass_TOGW.item()):.2f} kg")
            
            # Check convergence
            if abs(design_mass_TOGW - old_mass) < 0.01:  # 10g tolerance
                mass_converged = True
        
        # Solar power (exact from design_opt.py)
        solar_cell_efficiency = 0.243 * 0.9  # Sunpower
        latitude = 42.36  # Boston
        day_of_year = 91  # April 1
        time_after_solar_noon = 2 * 3600  # 2 hours in seconds
        wing_dihedral_angle_deg = 6
        
        power_in_at_panels = (
            0.5 * solar_area * lib_solar.solar_flux(
                latitude=latitude,
                day_of_year=day_of_year,
                time=time_after_solar_noon,
                altitude=cruise_op_point.atmosphere.altitude,
                panel_azimuth_angle=135 - 180,
                panel_tilt_angle=wing_dihedral_angle_deg,
            ) + 0.5 * solar_area * lib_solar.solar_flux(
                latitude=latitude,
                day_of_year=day_of_year,
                time=time_after_solar_noon,
                altitude=cruise_op_point.atmosphere.altitude,
                panel_azimuth_angle=135,
                panel_tilt_angle=wing_dihedral_angle_deg,
            )
        )
        
        MPPT_efficiency = 1 / 1.04
        power_in_total = power_in_at_panels * MPPT_efficiency * solar_cell_efficiency
        
        # Avionics power (exact from design_opt.py)
        avionics_power = 8  # Watts
        
        # Excess power for climbing
        power_out_total_cruise = cruise_power_propulsion + avionics_power
        excess_power_cruise = power_in_total - power_out_total_cruise
        
        # Calculate breakeven climb rate using converged mass
        if excess_power_cruise > 0 and design_mass_TOGW > 0:
            breakeven_climb_rate_mps = (
                excess_power_cruise * cruise_op_point.velocity /
                (cruise_power_propulsion * LD_cruise)
            )
            # Convert to ft/min
            breakeven_climb_rate_ftmin = breakeven_climb_rate_mps * 196.85  # m/s to ft/min
        else:
            breakeven_climb_rate_ftmin = 0.0
        # end 


        
        # Calculate constraints (exact from design_opt.py)
        # Constraint 1: mass closure - design mass must be greater than calculated mass
        constraint1 = design_mass_TOGW - design_mass_TOGW  # This will be 0, but keeping structure
        
        # Constraint 2: aerodynamics closure - LD_cruise must be less than 0.75 * LD_ideal
        # For simplicity, assume LD_ideal is 1.5x current LD (typical for ideal vs real)
        LD_ideal = LD_cruise * 1.5
        constraint2 = LD_cruise - 0.75 * LD_ideal
        
        # Constraint 3: CL must be positive
        constraint3 = -aero["CL"]  # Must be <= 0, so CL >= 0
        
        # Constraint 4: CL must be less than 0.8 (not near stall)
        constraint4 = aero["CL"] - 0.8
        
        # Constraint 5: airplane must lift itself (L > mg)
        lift_force = aero["L"]
        constraint5 = design_mass_TOGW * g - lift_force

        # Output 1: L/D 
        # output1 = aero["L"] / aero["D"]

        
        # Return [objective, constraint1, constraint2, constraint3, constraint4, constraint5]
        results.append([breakeven_climb_rate_ftmin, constraint1, constraint2, constraint3, constraint4, constraint5])
    

    # Convert back to torch.Tensor efficiently, return multi-output format
    results_array = np.array(results, dtype=np.float32)
    results_tensor = torch.from_numpy(results_array).to(device=x.device, dtype=torch.float32)
    
    # Ensure proper shape: (n_designs, 6_outputs)
    if results_tensor.dim() == 1:
        results_tensor = results_tensor.unsqueeze(0)
    
    return results_tensor

import numpy as np

def ground_roll_distance(V_R, W, T, D, L, mu, g=9.81):
    """
    Calculate ground roll distance for aircraft takeoff.
    
    Parameters:
    V_R : float
        Rotation velocity (m/s)
    W : float
        Aircraft weight (N)
    T : float
        Thrust (N)
    D : float
        Drag (N)
    L : float
        Lift (N)
    mu : float
        Ground friction coefficient
    g : float
        Gravitational acceleration (m/s²), default 9.81
    
    Returns:
    S_G : float
        Ground roll distance (m)
    """
    # Calculate the bracketed term at V_R/√2
    V_eval = V_R / np.sqrt(2)
    
    # The bracketed term: T - D - μ * (W - L)
    # Note: T, D, L should be evaluated at V_eval
    bracketed_term = T - D - mu * (W - L)
    
    # Calculate ground roll distance
    S_G = (V_R**2 * W) / (2 * g * bracketed_term)
    
    return S_G

# Example usage:
if __name__ == "__main__":
    # Example values (you'll need to provide actual values)
    V_R = 50.0  # m/s
    W = 10000   # N
    T = 5000    # N (at V_R/√2)
    D = 2000    # N (at V_R/√2)
    L = 8000    # N (at V_R/√2)
    mu = 0.02   # Ground friction coefficient
    
    S_G = ground_roll_distance(V_R, W, T, D, L, mu)
    print(f"Ground roll distance: {S_G:.2f} m")
    
def gen_op_points(design_mass_TOGW: float, CL_max: float, rho: float, S: float) -> float:
    """
    Compute the takeoff distance.
    """
    # Run aerodynamics analysis
    cruise_op_point = asb.OperatingPoint(
        velocity=14.0,  # m/s (from design_opt.py cruise speed)
        alpha=0.0,      # degrees (from design_opt.py cruise alpha)
        beta=0.0
    )

    aero = asb.AeroBuildup(
        airplane=airplane,
                op_point=cruise_op_point
    ).run()



    vs1 = np.sqrt(2 * design_mass_TOGW * GRAVITY / (0.5 * rho * S * CL_max))
    vr = 1.1 * vs1 # Gudmundsson Table 18-3
    return vs1

def train_gp_model(n_samples: int, bounds: list = None) -> SingleTaskGP:
    """
    Train a GP model by calling aerosandbox_objective n_samples times.
    
    Args:
        n_samples: Number of training samples to generate
        bounds: List of [min, max] for each design variable (optional)
    
    Returns:
        SingleTaskGP: Trained GP model
    """
    if not BOTORCH_AVAILABLE:
        print("BoTorch not available for GP training")
        return None
        
    if bounds is None:
        # Default bounds for span/chord
        bounds = [
            [5.0, 15.0],  # Span bounds
            [0.5, 2.0]    # Chord bounds
        ]
    
    print(f"Generating {n_samples} training samples...")
    
    # Generate random inputs in normalized space [0, 1]
    x_train = torch.rand(n_samples, len(bounds))
    
    # Call aerosandbox_objective_with_constraints to get y values
    print("Running AeroSandbox simulations...")
    y_train_full = aerosandbox_objective_with_constraints(x_train)
    
    # Extract only the objective (climb rate) for GP training
    # Negate climb rate so BO maximizes it (BO minimizes by default)
    y_train = y_train_full[:, 0:1].squeeze(-1)  # Shape: (n_samples, 1)
    
    print(f"Training data: X shape {x_train.shape}, y shape {y_train.shape}")
    print(f"Original climb rates: {y_train_full[:, 0].detach().cpu().numpy()}")
    print(f"Negated y values for GP: {y_train.detach().cpu().numpy()}")
    print(f"y range: {y_train.min():.3f} to {y_train.max():.3f}")
    
    # Train GP model
    print("Training GP model...")
    model = fit_single_task_gp(x_train, y_train)
    
    if model is not None:
        print("GP model trained successfully!")
    
    return model


def predict_with_gp(model: SingleTaskGP, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make predictions using the trained GP model."""
    if model is None:
        return None, None
        
    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean
        std = posterior.variance.sqrt()
    return mean, std


def bayesian_optimization_loop(
    n_initial: int = 1,
    n_iterations: int = 15,
    num_restarts: int = 50,  # Increased from 10
    raw_samples: int = 1024  # Increased from 512
):
    """
    Bayesian optimization loop to find the best span and chord values.
    
    Args:
        n_initial: Number of initial random samples
        n_iterations: Number of BO iterations
        num_restarts: Number of restarts for acquisition function optimization
        raw_samples: Number of raw samples for acquisition function optimization
    """
    if not BOTORCH_AVAILABLE:
        print("BoTorch not available for Bayesian optimization")
        return None, None, None, None
    
    print(f"=== Bayesian Optimization Loop ===")
    print(f"Initial samples: {n_initial}")
    print(f"BO iterations: {n_iterations}")
    
    # Define bounds for normalized space [0, 1]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    
    # Generate initial training data
    print(f"\nGenerating {n_initial} initial samples...")
    x_bo = torch.rand(n_initial, 2, dtype=torch.float32)
    y_bo_full = aerosandbox_objective_with_constraints(x_bo)
    
    # Extract only the objective (climb rate) for GP training
    # Negate climb rate so BO maximizes it (BO minimizes by default)
    y_bo = y_bo_full[:, 0:1].squeeze(-1)  # Shape: (n_initial, 1)
    
    print(f"Initial data: X shape {x_bo.shape}, y shape {y_bo.shape}")
    print(f"Initial original climb rates: {y_bo_full[:, 0].detach().cpu().numpy()}")
    print(f"Initial negated y values: {y_bo.detach().cpu().numpy()}")
    print(f"Initial y range: {y_bo.min():.3f} to {y_bo.max():.3f}")
    
    # Store best values
    best_values = []
    best_points = []
    
    # BO loop
    for i in range(n_iterations):
        print(f"\n--- BO Iteration {i+1}/{n_iterations} ---")
        
        # Fit GP model
        print(f"Fitting GP model {i+1}...")
        model = fit_single_task_gp(x_bo, y_bo)
        
        if model is None:
            print("Failed to fit GP model, skipping iteration")
            continue
        
        # Generate candidate using UCB acquisition function
        print(f"Generating candidate {i+1}...")
        from botorch.acquisition import UpperConfidenceBound
        acqf = UpperConfidenceBound(
            model,
            beta=2.0,  # Reduced exploration factor to avoid boundary sticking
        )
        
        candidates, value = optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_best_only=True,
            options={"maxiter": 1000},
        )
        
        suggested_point = candidates[0].detach().cpu().numpy()
        print(f"Suggested point: {suggested_point}")
        print(f"Acquisition function value: {value.item():.3f}")
        
        # Debug: Check if the suggested point is actually at a high UCB region
        with torch.no_grad():
            acqf_value_at_suggested = acqf(candidates).item()
            print(f"Acquisition function value at suggested point: {acqf_value_at_suggested:.3f}")
            
        # Debug: Check what the GP predicts at the suggested point
        with torch.no_grad():
            posterior_at_suggested = model.posterior(candidates)
            mean_at_suggested = posterior_at_suggested.mean.item()
            std_at_suggested = posterior_at_suggested.variance.sqrt().item()
            print(f"GP prediction at suggested point: mean={mean_at_suggested:.3f}, std={std_at_suggested:.3f}")
            print(f"UCB = mean + {2.0}*std = {mean_at_suggested + 2.0*std_at_suggested:.3f}")
        
        # Simulate the candidate
        print(f"Simulating candidate {i+1}...")
        y_new_full = aerosandbox_objective_with_constraints(candidates)
        
        # Extract only the objective (climb rate) for GP training
        # Negate climb rate so BO maximizes it (BO minimizes by default)
        y_new = y_new_full[:, 0:1].squeeze(-1)  # Shape: (1, 1)
        
        print(f"New original climb rate: {y_new_full[:, 0].item():.3f}")
        print(f"New negated objective value: {y_new[0, 0].item():.3f}")
        
        # Add new point to training data
        x_bo = torch.cat([x_bo, candidates], dim=0)
        y_bo = torch.cat([y_bo, y_new], dim=0)
        
        # Track best value (remember y_bo is now negative, so we minimize)
        best_idx = y_bo.argmax()  # argmin because we negated the values
        best_value = y_bo[best_idx].item()  # Convert back to positive for display
        best_point = x_bo[best_idx].detach().cpu().numpy()
        
        best_values.append(best_value)
        best_points.append(best_point)
        
        # Denormalize best point to physical values
        span_bounds = [5.0, 15.0]
        chord_bounds = [0.5, 2.0]
        best_span = best_point[0] * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
        best_chord = best_point[1] * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]
        
        print(f"Current best: span={best_span:.2f}m, chord={best_chord:.2f}m, Max Climb Rate={best_value:.1f} ft/min")
        print(f"Training data size: {len(x_bo)}")
        
        # Create visualization for every iteration
        print(f"\nCreating visualization for iteration {i+1}...")
        create_bo_visualization(x_bo, y_bo, model, i+1, bounds, span_bounds, chord_bounds)
    
    print(f"\n=== BO Complete ===")
    print(f"Final best: span={best_span:.2f}m, chord={best_chord:.2f}m, Max Climb Rate={best_value:.1f} ft/min")
    
    return x_bo, y_bo, best_values, best_points


def create_bo_visualization(
    x_bo: torch.Tensor,
    y_bo: torch.Tensor,
    model: SingleTaskGP,
    iteration: int,
    bounds: torch.Tensor,
    span_bounds: list = [5.0, 15.0],
    chord_bounds: list = [0.5, 2.0]
):
    """
    Create visualization plots for Bayesian optimization progress.
    
    Args:
        x_bo: Training inputs (normalized)
        y_bo: Training outputs (climb rates)
        model: Trained GP model
        iteration: Current iteration number
        bounds: Normalized bounds for the design space
        span_bounds: Physical bounds for span
        chord_bounds: Physical bounds for chord
    """
    if not BOTORCH_AVAILABLE:
        return
        
    # Create a grid for visualization
    n_grid = 50
    x1_grid = torch.linspace(bounds[0, 0], bounds[1, 0], n_grid)
    x2_grid = torch.linspace(bounds[0, 1], bounds[1, 1], n_grid)
    X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing='ij')
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    
    # Get GP predictions
    with torch.no_grad():
        posterior = model.posterior(X_grid)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
        
        # Calculate UCB acquisition function (same as used in BO)
        ucb = mean + 2.0 * std  # Match the beta value used in optimization  # Same beta=3.0 as in BO
        ucb = ucb.reshape(n_grid, n_grid)
        
        mean = mean.reshape(n_grid, n_grid)
        std = std.reshape(n_grid, n_grid)
    
    # Convert to numpy for plotting
    X1_np = X1.detach().cpu().numpy()
    X2_np = X2.detach().cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()
    ucb_np = ucb.detach().cpu().numpy()
    
    # Convert GP predictions back to positive climb rates for visualization
    #mean_np = -mean_np  # Convert from negated back to positive
    #ucb_np = -ucb_np    # Convert from negated back to positive
    
    # Denormalize for physical coordinates
    span_phys = X1_np * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
    chord_phys = X2_np * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]
    
    # Convert training data to physical coordinates
    x_bo_phys = x_bo.detach().cpu().numpy()
    span_train = x_bo_phys[:, 0] * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
    chord_train = x_bo_phys[:, 1] * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]
    # Convert back to positive values for visualization
    y_bo_np = -y_bo.detach().cpu().numpy().flatten()
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: GP Mean Prediction
    im1 = ax1.contourf(span_phys, chord_phys, mean_np, levels=20, cmap='viridis')
    # Plot all points except the newest one in black
    if len(span_train) > 1:
        scatter1 = ax1.scatter(span_train[:-1], chord_train[:-1], c='black', s=100, 
                    edgecolors='white', linewidth=2, zorder=5)
    # Plot the newest point in white
    scatter1_new = ax1.scatter(span_train[-1], chord_train[-1], c='white', s=150, 
                edgecolors='white', linewidth=2, zorder=6)
    ax1.set_xlabel('Wing Span (m)')
    ax1.set_ylabel('Mean Chord (m)')
    ax1.set_title(f'GP Mean Prediction - Iteration {iteration}')
    plt.colorbar(im1, ax=ax1, label='Climb Rate (ft/min)')
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: Acquisition Function (UCB)
    im2 = ax2.contourf(span_phys, chord_phys, ucb_np, levels=20, cmap='plasma')
    # Plot all points except the newest one in red
    if len(span_train) > 1:
        ax2.scatter(span_train[:-1], chord_train[:-1], c='red', s=100, marker='o', 
                    edgecolors='white', linewidth=2, zorder=5, label='Training Points')
    # Plot the newest point in white
    ax2.scatter(span_train[-1], chord_train[-1], c='white', s=150, 
                edgecolors='white', linewidth=2, zorder=6, label='New Point')
    ax2.set_xlabel('Wing Span (m)')
    ax2.set_ylabel('Mean Chord (m)')
    ax2.set_title(f'Acquisition Function (UCB) - Iteration {iteration}')
    plt.colorbar(im2, ax=ax2, label='UCB Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)  # Don't block execution
    plt.pause(2)  # Show for 2 seconds
    plt.close()  # Close automatically
    
    # Print current best point
    best_idx = y_bo_np.argmax()  # y_bo_np is already positive
    best_span = span_train[best_idx]
    best_chord = chord_train[best_idx]
    best_climb = y_bo_np[best_idx]
    
    print(f"Current best: span={best_span:.2f}m, chord={best_chord:.2f}m, climb_rate={best_climb:.1f} ft/min")


if __name__ == "__main__":
    # Test the wrapper
    print("=== Testing AeroSandbox Wrapper ===")
    
    # Test single design
    x = torch.tensor([[0, 1]])  # Single design, normalized [0.5, 0.5]
    result = aerosandbox_objective_with_constraints(x)
    print(f"Single result: {result}")
    print(f"  Climb rate: {result[0, 0].item():.1f} ft/min")
    print(f"  Constraints: {result[0, 1:].numpy()}")
    
    # Test multiple designs
    x = torch.tensor([[0.2, 0.3], [0.8, 0.7], [0.5, 0.5]])
    result = aerosandbox_objective_with_constraints(x)
    print(f"Multiple results: {result}")
    print(f"  Climb rates: {result[:, 0].numpy()}")
    print(f"  Constraint violations: {result[:, 1:].numpy()}")

    plot_doe = False

    if plot_doe:
        
        # Complete DOE to map the entire objective function landscape
        print("\n=== Complete Design of Experiments ===")
        print("Generating meshgrid of all possible span/chord combinations...")
        
        # Create a fine meshgrid
        n_grid = 10  # 10x10 = 100 total evaluations
        span_norm = torch.linspace(0, 1, n_grid)
        chord_norm = torch.linspace(0, 1, n_grid)
        span_mesh, chord_mesh = torch.meshgrid(span_norm, chord_norm, indexing='ij')
        
        # Flatten to get all combinations
        span_flat = span_mesh.flatten()
        chord_flat = chord_mesh.flatten()
        x_doe = torch.stack([span_flat, chord_flat], dim=1)
        
        print(f"Evaluating {len(x_doe)} design points...")
        
        # Evaluate all points
        results_doe = aerosandbox_objective_with_constraints(x_doe)
        climb_rates_doe = results_doe[:, 0].detach().cpu().numpy()
        
        # Find the best point
        best_idx_doe = np.argmax(climb_rates_doe)
        best_span_norm = span_flat[best_idx_doe].item()
        best_chord_norm = chord_flat[best_idx_doe].item()
        best_climb_doe = climb_rates_doe[best_idx_doe]
        
        # Convert to physical coordinates
        span_bounds = [5.0, 15.0]
        chord_bounds = [0.5, 2.0]
        best_span_phys = best_span_norm * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
        best_chord_phys = best_chord_norm * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]
        
        print(f"Best point found: span={best_span_phys:.2f}m, chord={best_chord_phys:.2f}m")
        print(f"Best climb rate: {float(best_climb_doe):.1f} ft/min")
        print(f"Normalized coordinates: [{best_span_norm:.3f}, {best_chord_norm:.3f}]")
        
        # Create the DOE visualization
        print("\nCreating DOE visualization...")
        import matplotlib.pyplot as plt
        
        # Reshape for plotting
        span_phys = span_mesh.detach().cpu().numpy() * (span_bounds[1] - span_bounds[0]) + span_bounds[0]
        chord_phys = chord_mesh.detach().cpu().numpy() * (chord_bounds[1] - chord_bounds[0]) + chord_bounds[0]
        climb_rates_plot = climb_rates_doe.reshape(n_grid, n_grid)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot the color map
        im = ax.contourf(span_phys, chord_phys, climb_rates_plot, levels=20, cmap='viridis')
        
        # Mark the best point
        ax.scatter(best_span_phys, best_chord_phys, c='red', s=200, marker='*', 
                edgecolors='white', linewidth=2, zorder=10, label=f'Best: {float(best_climb_doe):.0f} ft/min')
        
        # Add colorbar and labels
        plt.colorbar(im, ax=ax, label='Climb Rate (ft/min)')
        ax.set_xlabel('Wing Span (m)')
        ax.set_ylabel('Mean Chord (m)')
        ax.set_title('Complete DOE: Climb Rate Landscape')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    # end 
    
    # Train GP model
    print("\n=== Training GP Model ===")
    gp_model = train_gp_model(n_samples=20)
    
    if gp_model is not None:
        # Test predictions
        test_x = torch.tensor([[0.6, 0.4], [0.3, 0.8]])
        pred_mean, pred_std = predict_with_gp(gp_model, test_x)
        
        if pred_mean is not None:
            print("\nTest predictions:")
            for i, (x, pred, std) in enumerate(zip(test_x, pred_mean, pred_std)):
                print(f"  Point {i+1}: Pred={pred[0]:.3f} ± {std[0]:.3f}")
    
    # Run Bayesian optimization
    print("\n=== Running Bayesian Optimization ===")
    x_bo, y_bo, best_values, best_points = bayesian_optimization_loop(
        n_initial=5,  # Start with more diverse initial points
        n_iterations=15  # More iterations to explore better
    )
    
    print("\n=== Done ===")
