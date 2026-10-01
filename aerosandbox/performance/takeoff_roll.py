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
from aerosandbox.library import propulsion_turbofan as lib_prop_turbofan
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

def aircraft_setup():

    ac = {}
    ac.s_ref = 100
    ac.c_ref = 10
    ac.b_ref = 100
    ac.wings = [wing]
    return ac



def kinematics_setup():
    kin = {}
    kin.velocity = 100
    kin.alpha = 0
    kin.beta = 0
    return kin

def main():

    span = 100
    chord = 10
    # Create a complete airplane similar to design_opt.py
    # Basic configuration
    x_tail = 0 + span * 0.38
    
    # Wing with proper airfoils and twist
    wing = asb.Wing(
        name="Wing",
            symmetric=True,
            xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=chord,
                twist=4.70,
                airfoil=asb.Airfoil("ag34")
            ),
            asb.WingXSec(
                xyz_le=[0, span / 2 * 0.6, 0],  # 60% break point
                chord=chord,
                twist=3.50,
                airfoil=asb.Airfoil("ag34")
            ),
            asb.WingXSec(
                xyz_le=[0, span / 2, 0],
                chord=chord * 0.8,  # Tapered tip
                twist=3.00,
                airfoil=asb.Airfoil("ag36")
            )
        ]
    )
    
    # Create complete airplane
    airplane = asb.Airplane(
        name="Z4",
        wings=[wing]
    )
    
    # Run aerodynamics analysis
    cruise_op_point = asb.OperatingPoint(
        velocity=100.0,  # m/s (from design_opt.py cruise speed)
        alpha=0.0,      # degrees (from design_opt.py cruise alpha)
        beta=0.0
    )
        
    aero = asb.AeroBuildup(
        airplane=airplane,
                    op_point=cruise_op_point
    ).run()
    
    # Get L/D ratio
    LD_cruise = aero["CL"] / aero["CD"]
    print(LD_cruise)



if __name__ == "__main__":

    main()