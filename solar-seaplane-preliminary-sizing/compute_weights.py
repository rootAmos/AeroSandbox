from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from computeptrainweights import HybridPowertrainWeight
import pdb
import matplotlib.pyplot as plt
from formatmassresults import FormatMassResults

class WingWeight(om.ExplicitComponent):
    """Computes wing weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_mto", val=1.0, units="lbf", desc="Maximum takeoff weight")
        self.add_input("n_ult", val=1.0, units=None, desc="Ultimate load factor")
        self.add_input("s_ref", val=1.0, units="ft**2", desc="Wing area")
        self.add_input("ar_w", val=1.0, units=None, desc="Wing aspect ratio")
        self.add_input("t_c_w", val=1.0, units=None, desc="Wing thickness-to-chord ratio")
        self.add_input("sweep_c_4_w", val=1.0, units="rad", desc="Wing quarter-chord sweep angle")
        self.add_input("q_cruise", val=1.0, units="lbf/ft**2", desc="Cruise dynamic pressure")
        self.add_input("lambda_w", val=1.0, units=None, desc="Wing taper ratio")
        self.add_input("w_fuel", val=1.0, units="lbf", desc="Weight of fuel in wings")
        
        self.add_output("w_wing", units="lbf",
                       desc="Wing weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        w_mto = inputs["w_mto"] 
        n_ult = inputs["n_ult"]
        s_ref = inputs["s_ref"]
        ar_w = inputs["ar_w"]
        t_c_w = inputs["t_c_w"]
        sweep_c_4_w = inputs["sweep_c_4_w"]
        q_cruise = inputs["q_cruise"]
        lambda_w = inputs["lambda_w"]
        w_fuel = inputs["w_fuel"]
        

        outputs["w_wing"] = (
        0.036 
        * s_ref**0.758 
        * w_fuel**0.0035 
        * (ar_w / (np.cos(sweep_c_4_w)**2))**0.6 
        * q_cruise**0.006 
        * lambda_w**0.04 
        * ((100 * t_c_w) / np.cos(sweep_c_4_w))**-0.3 
        * (n_ult * w_mto)**0.49
    )
        
    def compute_partials(self, inputs, partials):   

        w_mto = inputs["w_mto"]
        n_ult = inputs["n_ult"]
        s_ref = inputs["s_ref"]
        ar_w = inputs["ar_w"]
        t_c_w = inputs["t_c_w"]
        sweep_c_4_w = inputs["sweep_c_4_w"]
        q_cruise = inputs["q_cruise"]
        lambda_w = inputs["lambda_w"]
        w_fuel = inputs["w_fuel"]


        partials["w_wing", "s_ref"] = (0.0273*lambda_w**0.0400*q_cruise**0.0060*w_fuel**0.0035*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(s_ref**0.2420*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "w_fuel"] =(1.2600e-04*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(w_fuel**0.9965*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "w_mto"] = (0.0176*lambda_w**0.0400*n_ult*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/((n_ult*w_mto)**0.5100*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "ar_w"] = (0.0216*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*(n_ult*w_mto)**0.4900)/(np.cos(sweep_c_4_w)**2*(ar_w/np.cos(sweep_c_4_w)**2)**0.4000*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "q_cruise"] =(2.1600e-04*lambda_w**0.0400*s_ref**0.7580*w_fuel**0.0035*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(q_cruise**0.9940*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "lambda_w"] = (0.0014*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(lambda_w**0.9600*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "n_ult"] = (0.0176*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*w_mto*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/((n_ult*w_mto)**0.5100*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000)

        partials["w_wing", "t_c_w"] = -(1.0800*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(np.cos(sweep_c_4_w)*((100*t_c_w)/np.cos(sweep_c_4_w))**1.3000)

        partials["w_wing", "sweep_c_4_w"] = (0.0432*ar_w*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*w_fuel**0.0035*np.sin(sweep_c_4_w)*(n_ult*w_mto)**0.4900)/(np.cos(sweep_c_4_w)**3*(ar_w/np.cos(sweep_c_4_w)**2)**0.4000*((100*t_c_w)/np.cos(sweep_c_4_w))**0.3000) - (1.0800*lambda_w**0.0400*q_cruise**0.0060*s_ref**0.7580*t_c_w*w_fuel**0.0035*np.sin(sweep_c_4_w)*(n_ult*w_mto)**0.4900*(ar_w/np.cos(sweep_c_4_w)**2)**0.6000)/(np.cos(sweep_c_4_w)**2*((100*t_c_w)/np.cos(sweep_c_4_w))**1.3000)

        

class HorizontalTailWeight(om.ExplicitComponent):
    """Computes horizontal tail weight using Raymer's equation."""
    
    def setup(self):
        # Add inputs
        self.add_input('n_ult', val=1.0, units=None, desc='Ultimate load factor')
        self.add_input('w_mto', val=1.0, units='lbf', desc='Takeoff gross weight')
        self.add_input('q_cruise', val=1.0, units='lbf/ft**2', desc='Dynamic pressure at cruise')
        self.add_input('s_ht', val=1.0, units='ft**2', desc='Horizontal tail area')
        self.add_input('t_c_ht', val=1.0, units=None, desc='Thickness-to-chord ratio')
        self.add_input('sweep_c_4_ht', val=1.0, units='rad', desc='Horizontal tail quarter-chord sweep angle')
        self.add_input('ar_w', val=1.0, units=None, desc='Wing aspect ratio')
        self.add_input('lambda_ht', val=1.0, units=None, desc='Horizontal tail taper ratio')
        
        # Add output
        self.add_output('w_htail', units='lbf',
                       desc='Horizontal tail weight')
        
        self.declare_partials('*', '*', method='exact')
        
    def compute(self, inputs, outputs):
        n_ult = inputs['n_ult']
        w_mto = inputs['w_mto']
        q_cruise = inputs['q_cruise']
        s_ht = inputs['s_ht']
        t_c_ht = inputs['t_c_ht']
        sweep_c_4_ht = inputs['sweep_c_4_ht']
        ar_w = inputs['ar_w']
        lambda_ht = inputs['lambda_ht']
        
        outputs['w_htail'] = (
            0.016 
            * (n_ult * w_mto)**0.414
            * q_cruise**0.168
            * s_ht**0.896 
            * ((100 * t_c_ht) / np.cos(sweep_c_4_ht))**-0.12
            * (ar_w / (np.cos(sweep_c_4_ht)**2))**0.043
            * lambda_ht**-0.02
        )

    def compute_partials(self, inputs, partials):

        n_ult = inputs["n_ult"]
        w_mto = inputs["w_mto"]
        q_cruise = inputs["q_cruise"]
        s_ht = inputs["s_ht"]
        t_c_ht = inputs["t_c_ht"]
        sweep_c_4_ht = inputs["sweep_c_4_ht"]
        ar_w = inputs["ar_w"]
        lambda_ht = inputs["lambda_ht"]

        partials["w_htail", "n_ult"] = (0.0066*q_cruise**0.1680*s_ht**0.8960*w_mto*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*(n_ult*w_mto)**0.5860*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)

        partials["w_htail", "w_mto"] = (0.0066*n_ult*q_cruise**0.1680*s_ht**0.8960*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*(n_ult*w_mto)**0.5860*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)

        partials["w_htail", "q_cruise"] = (0.0027*s_ht**0.8960*(n_ult*w_mto)**0.4140*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*q_cruise**0.8320*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)

        partials["w_htail", "s_ht"] = (0.0143*q_cruise**0.1680*(n_ult*w_mto)**0.4140*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*s_ht**0.1040*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)

        partials["w_htail", "t_c_ht"] = -(0.1920*q_cruise**0.1680*s_ht**0.8960*(n_ult*w_mto)**0.4140*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*np.cos(sweep_c_4_ht)*((100*t_c_ht)/np.cos(sweep_c_4_ht))**1.1200)

        partials["w_htail", "sweep_c_4_ht"] = (0.0014*ar_w*q_cruise**0.1680*s_ht**0.8960*np.sin(sweep_c_4_ht)*(n_ult*w_mto)**0.4140)/(lambda_ht**0.0200*np.cos(sweep_c_4_ht)**3*(ar_w/np.cos(sweep_c_4_ht)**2)**0.9570*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200) - (0.1920*q_cruise**0.1680*s_ht**0.8960*t_c_ht*np.sin(sweep_c_4_ht)*(n_ult*w_mto)**0.4140*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**0.0200*np.cos(sweep_c_4_ht)**2*((100*t_c_ht)/np.cos(sweep_c_4_ht))**1.1200)

        partials["w_htail", "ar_w"] = (6.8800e-04*q_cruise**0.1680*s_ht**0.8960*(n_ult*w_mto)**0.4140)/(lambda_ht**0.0200*np.cos(sweep_c_4_ht)**2*(ar_w/np.cos(sweep_c_4_ht)**2)**0.9570*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)

        partials["w_htail", "lambda_ht"] = -(3.2000e-04*q_cruise**0.1680*s_ht**0.8960*(n_ult*w_mto)**0.4140*(ar_w/np.cos(sweep_c_4_ht)**2)**0.0430)/(lambda_ht**1.0200*((100*t_c_ht)/np.cos(sweep_c_4_ht))**0.1200)



class VerticalTailWeight(om.ExplicitComponent):
    """
    Computes vertical tail weight using Raymer's equation.
    """
    
    def setup(self):
        # Add inputs
        self.add_input('f_tail', val=1.0, units=None, desc='tail shape factor. 0 for conventional tail, 1 for t-tail')
        self.add_input('n_ult', val=1.0, units=None, desc='Ultimate load factor')
        self.add_input('w_mto', val=1.0, units='lbf', desc='Takeoff gross weight')
        self.add_input('q_cruise', val=1.0, units='lbf/ft**2', desc='Dynamic pressure at cruise')
        self.add_input('s_vt', val=1.0, units='ft**2', desc='Vertical tail area')
        self.add_input('t_c_vt', val=1.0, units=None, desc='Thickness-to-chord ratio')
        self.add_input('sweep_c_4_vt', val=1.0, units='rad', desc='Vertical tail quarter-chord sweep angle')
        self.add_input('ar_w', val=1.0, units=None, desc='Wing aspect ratio')
        self.add_input('lambda_vt', val=1.0, units=None, desc='Vertical tail taper ratio')
        
        # Add output
        self.add_output('w_vtail', units='lbf',
                       desc='Vertical tail weight')
        
        self.declare_partials('*', '*', method='exact')
        
    def compute(self, inputs, outputs):

        f_tail = inputs['f_tail']
        n_ult = inputs['n_ult']
        w_mto = inputs['w_mto']
        q_cruise = inputs['q_cruise']
        s_vt = inputs['s_vt']
        t_c_vt = inputs['t_c_vt']
        sweep_c_4_vt = inputs['sweep_c_4_vt']
        ar_w = inputs['ar_w']
        lambda_vt = inputs['lambda_vt']
        
        outputs['w_vtail'] = (
            0.073
            * (1 + 0.2 * f_tail)
            * (n_ult * w_mto)**0.376
            * q_cruise**0.122
            * s_vt**0.873
            * ((100 * t_c_vt) / np.cos(sweep_c_4_vt))**-0.49
            * (ar_w / (np.cos(sweep_c_4_vt)**2))**0.357
            * lambda_vt**0.039
        )

    def compute_partials(self, inputs, partials):
    

        n_ult = inputs["n_ult"]
        w_mto = inputs["w_mto"]
        q_cruise = inputs["q_cruise"]
        s_vt = inputs["s_vt"]
        t_c_vt = inputs["t_c_vt"]
        sweep_c_4_vt = inputs["sweep_c_4_vt"]
        ar_w = inputs["ar_w"]
        lambda_vt = inputs["lambda_vt"]
        f_tail = inputs["f_tail"]




        partials["w_vtail", "n_ult"] = (0.3760*lambda_vt**0.0390*q_cruise**0.1220*s_vt**0.8730*w_mto*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/((n_ult*w_mto)**0.6240*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        partials["w_vtail", "w_mto"] = (0.3760*lambda_vt**0.0390*n_ult*q_cruise**0.1220*s_vt**0.8730*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/((n_ult*w_mto)**0.6240*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        partials["w_vtail", "q_cruise"] = (0.1220*lambda_vt**0.0390*s_vt**0.8730*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/(q_cruise**0.8780*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        partials["w_vtail", "s_vt"] = (0.8730*lambda_vt**0.0390*q_cruise**0.1220*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/(s_vt**0.1270*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        partials["w_vtail", "t_c_vt"] = -(49*lambda_vt**0.0390*q_cruise**0.1220*s_vt**0.8730*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/(np.cos(sweep_c_4_vt)*((100*t_c_vt)/np.cos(sweep_c_4_vt))**1.4900)

        partials["w_vtail", "sweep_c_4_vt"] = (0.7140*ar_w*lambda_vt**0.0390*q_cruise**0.1220*s_vt**0.8730*np.sin(sweep_c_4_vt)*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730))/(np.cos(sweep_c_4_vt)**3*(ar_w/np.cos(sweep_c_4_vt)**2)**0.6430*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900) - (49*lambda_vt**0.0390*q_cruise**0.1220*s_vt**0.8730*t_c_vt*np.sin(sweep_c_4_vt)*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/(np.cos(sweep_c_4_vt)**2*((100*t_c_vt)/np.cos(sweep_c_4_vt))**1.4900)

        partials["w_vtail", "ar_w"] = (0.3570*lambda_vt**0.0390*q_cruise**0.1220*s_vt**0.8730*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730))/(np.cos(sweep_c_4_vt)**2*(ar_w/np.cos(sweep_c_4_vt)**2)**0.6430*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        partials["w_vtail", "lambda_vt"] = (0.0390*q_cruise**0.1220*s_vt**0.8730*(n_ult*w_mto)**0.3760*(0.0146*f_tail + 0.0730)*(ar_w/np.cos(sweep_c_4_vt)**2)**0.3570)/(lambda_vt**0.9610*((100*t_c_vt)/np.cos(sweep_c_4_vt))**0.4900)

        

class ElectricalSystemWeight(om.ExplicitComponent):
    """Computes electrical system weight using Raymer/USAF equation."""
    
    def setup(self):
        self.add_input("w_fuel_system", val=1.0, units="lbf", desc="Fuel system weight")
        self.add_input("w_avionics", val=1.0, units="lbf", desc="Avionics system weight")
        
        self.add_output("w_electrical", units="lbf",
                       desc="Electrical system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        w_fuel_system = inputs["w_fuel_system"]
        w_avionics = inputs["w_avionics"]
        
        outputs["w_electrical"] = 12.57 * (w_fuel_system + w_avionics)**0.51

    def compute_partials(self, inputs, partials):
        w_fuel_system = inputs["w_fuel_system"]
        w_avionics = inputs["w_avionics"]
        
        partials["w_electrical", "w_fuel_system"] = 12.57 * 0.51 * (w_fuel_system + w_avionics)**-0.49
        partials["w_electrical", "w_avionics"] = 12.57 * 0.51 * (w_fuel_system + w_avionics)**-0.49
        

class FurnishingsWeight(om.ExplicitComponent):
    """Computes furnishings weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_mto", val=1.0, units="lbf", desc="Takeoff gross weight")
        
        self.add_output("w_furnishings", units="lbf",
                       desc="Furnishings weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        
        w_mto = inputs["w_mto"]
        outputs["w_furnishings"] = 0.0582 * w_mto - 65

    def compute_partials(self, inputs, partials):

        partials["w_furnishings", "w_mto"] = 0.0582


    
class AvionicsWeight(om.ExplicitComponent):
    """Computes avionics system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_uav", val=1.0, units="lbf", desc="Uninstalled avionics weight")
        
        self.add_output("w_avionics", units="lbf",
                       desc="Installed avionics system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        w_uav = inputs["w_uav"]
        outputs["w_avionics"] = 2.11 * w_uav**0.933

    def compute_partials(self, inputs, partials):

        w_uav = inputs["w_uav"]
        partials["w_avionics", "w_uav"] = 2.11 * 0.933 * w_uav**-0.067
        

class FlightControlSystemWeight(om.ExplicitComponent):
    """Computes flight control system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("l_fs", val=1.0, units="ft", desc="Fuselage length")
        self.add_input("b_w", val=1.0, units="ft", desc="Wingspan")
        self.add_input("n_ult", val=1.0, units=None, desc="Ultimate load factor")
        self.add_input("w_mto", val=1.0, units="lbf", desc="Takeoff gross weight")
        
        self.add_output("w_fcs", units="lbf",
                       desc="Flight control system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):


        l_fs = inputs["l_fs"]
        b_w = inputs["b_w"]
        n_ult = inputs["n_ult"]
        w_mto = inputs["w_mto"]
        
        outputs["w_fcs"] = (
            0.053 
            * l_fs**1.536 
            * b_w**0.371 
            * ((n_ult * w_mto * 1e-4)**0.80)
        )

    def compute_partials(self, inputs, partials):

        l_fs = inputs["l_fs"]
        b_w = inputs["b_w"]
        n_ult = inputs["n_ult"]
        w_mto = inputs["w_mto"]
        
        partials["w_fcs", "l_fs"] =0.0814*b_w**0.3710*l_fs**0.5360*(1.0000e-04*n_ult*w_mto)**0.8000

        partials["w_fcs", "b_w"] = (0.0197*l_fs**1.5360*(1.0000e-04*n_ult*w_mto)**0.8000)/b_w**0.6290

        partials["w_fcs", "n_ult"] = (4.2400e-06*b_w**0.3710*l_fs**1.5360*w_mto)/(1.0000e-04*n_ult*w_mto)**0.2000

        partials["w_fcs", "w_mto"] = (4.2400e-06*b_w**0.3710*l_fs**1.5360*n_ult)/(1.0000e-04*n_ult*w_mto)**0.2000


                      

        

class FuselageWeight(om.ExplicitComponent):
    """Computes fuselage weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_mto", val=1.0, units="lbf", desc="Maximum takeoff weight")
        self.add_input("n_ult", val=1.0, units=None, desc="Ultimate load factor")
        self.add_input("q_cruise", val=1.0, units="lbf/ft**2", desc="Dynamic pressure at cruise")
        self.add_input("l_fs", val=1.0, units="ft", desc="Fuselage structural length")
        self.add_input("d_fs", val=1.0, units="ft", desc="Fuselage structural diameter")
        self.add_input("l_ht", val=1.0, units="ft", desc="Horizontal tail lever arm length")
        self.add_input("v_fp", val=1.0, units="ft**3", desc="Fuselage pressurized volume")
        self.add_input("delta_p", val=1.0, units="psi", desc="Fuselage pressurization delta")
        self.add_input("s_fus", val=1.0, units="ft**2", desc="Fuselage wetted area")
        
        self.add_output("w_fuselage", units="lbf",
                       desc="Fuselage weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        w_mto = inputs["w_mto"]
        n_ult = inputs["n_ult"]
        l_fs = inputs["l_fs"]
        d_fs = inputs["d_fs"]
        l_ht = inputs["l_ht"]
        v_fp = inputs["v_fp"]
        delta_p = inputs["delta_p"]
        s_fus = inputs["s_fus"]
        q_cruise = inputs["q_cruise"]
        
        outputs["w_fuselage"] = 0.052 * s_fus **1.086 * ( n_ult * w_mto) ** 0.177 * \
            l_ht ** -0.051 *(l_fs/d_fs) ** -0.072 *  q_cruise ** 0.241 + 11.9 *(v_fp * delta_p) **0.271
        
    def compute_partials(self, inputs, partials):

        # Unpack inputs
        w_mto = inputs["w_mto"]
        n_ult = inputs["n_ult"]
        l_fs = inputs["l_fs"]
        d_fs = inputs["d_fs"]
        l_ht = inputs["l_ht"]
        v_fp = inputs["v_fp"]
        delta_p = inputs["delta_p"]
        s_fus = inputs["s_fus"]
        q_cruise = inputs["q_cruise"]

        partials["w_fuselage", "w_mto"] = (0.0092*n_ult*q_cruise**0.2410*s_fus**1.0860)/(l_ht**0.0510*(n_ult*w_mto)**0.8230*(l_fs/d_fs)**0.0720)

        partials["w_fuselage", "n_ult"] = (0.0092*q_cruise**0.2410*s_fus**1.0860*w_mto)/(l_ht**0.0510*(n_ult*w_mto)**0.8230*(l_fs/d_fs)**0.0720)

        partials["w_fuselage", "l_fs"] = -(0.0037*q_cruise**0.2410*s_fus**1.0860*(n_ult*w_mto)**0.1770)/(d_fs*l_ht**0.0510*(l_fs/d_fs)**1.0720)

        partials["w_fuselage", "d_fs"] = (0.0037*l_fs*q_cruise**0.2410*s_fus**1.0860*(n_ult*w_mto)**0.1770)/(d_fs**2*l_ht**0.0510*(l_fs/d_fs)**1.0720)

        partials["w_fuselage", "l_ht"] = -(0.0027*q_cruise**0.2410*s_fus**1.0860*(n_ult*w_mto)**0.1770)/(l_ht**1.0510*(l_fs/d_fs)**0.0720)

        partials["w_fuselage", "v_fp"] = (3.2249*delta_p)/(delta_p*v_fp)**0.7290

        partials["w_fuselage", "delta_p"] = (3.2249*v_fp)/(delta_p*v_fp)**0.7290

        partials["w_fuselage", "s_fus"] = (0.0565*q_cruise**0.2410*s_fus**0.0860*(n_ult*w_mto)**0.1770)/(l_ht**0.0510*(l_fs/d_fs)**0.0720)

        partials["w_fuselage", "q_cruise"] = (0.0125*s_fus**1.0860*(n_ult*w_mto)**0.1770)/(l_ht**0.0510*q_cruise**0.7590*(l_fs/d_fs)**0.0720)




class LandingGearWeight(om.ExplicitComponent):
    """Computes landing gear weight. Ref Gudmundsson Eq 6-61 & 6-66."""
    
    def initialize(self):
        self.options.declare("A_main", default=20.0,    
                      desc="Main landing gear A constant")
        self.options.declare("A_nose", default=25.0,
                      desc="Nose landing gear A constant")
        self.options.declare("B_main", default=0.1,
                      desc="Main landing gear B constant")
        self.options.declare("B_nose", default=0,
                      desc="Nose landing gear B constant")
        self.options.declare("C_main", default=0.019,
                      desc="Main landing gear C constant")
        self.options.declare("C_nose", default=0.0024,
                      desc="Nose landing gear C constant")
        self.options.declare("D_main", default=0,
                      desc="Main landing gear D constant")
        self.options.declare("D_nose", default=0,
                      desc="Nose landing gear D constant")
        
    def setup(self):
        self.add_input("w_mto", val=1.0, units="lbf", desc="Design gross weight")
        self.add_input("w_fuel", val=1.0, units="lbf", desc="fuel weight")
        self.add_input("n_l", val=1.0, units=None, desc="Ultimate landing load factor")
        self.add_input("l_m", val=1.0, units="ft", desc="Length of main landing gear strut")
        self.add_input("l_n", val=1.0, units="ft", desc="Length of nose landing gear strut")
        
        self.add_output("w_lg", units="lbf",
                       desc="total landing gear weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        w_mto = inputs["w_mto"]
        w_fuel = inputs["w_fuel"]
        n_l = inputs["n_l"]
        l_m = inputs["l_m"]
        l_n = inputs["l_n"]

        # Main landing gear weight

        w_mlg = 0.095 * (n_l * (w_mto - w_fuel) ** 0.768) * (l_m ** 0.409)

        # Nose landing gear weight
        w_nlg = 0.125 * (n_l * (w_mto - w_fuel) ** 0.566) * (l_n ** 0.845)

        outputs["w_lg"] = (w_mlg + w_nlg) 

    def compute_partials(self, inputs, partials):

        # Unpack inputs
        w_mto = inputs["w_mto"]
        w_fuel = inputs["w_fuel"]
        n_l = inputs["n_l"]
        l_m = inputs["l_m"]
        l_n = inputs["l_n"]

        partials["w_lg", "w_mto"] = (0.0708*l_n**0.8450*n_l)/(w_mto - w_fuel)**0.4340 + (0.0730*l_m**0.4090*n_l)/(w_mto - w_fuel)**0.2320

        partials["w_lg", "w_fuel"] = - (0.0708*l_n**0.8450*n_l)/(w_mto - w_fuel)**0.4340 - (0.0730*l_m**0.4090*n_l)/(w_mto - w_fuel)**0.2320

        partials["w_lg", "l_m"] = (0.0389*n_l*(w_mto - w_fuel)**0.7680)/l_m**0.5910

        partials["w_lg", "l_n"] = (0.1056*n_l*(w_mto - w_fuel)**0.5660)/l_n**0.1550

        partials["w_lg", "n_l"] = 0.0950*l_m**0.4090*(w_mto - w_fuel)**0.7680 + 0.1250*l_n**0.8450*(w_mto - w_fuel)**0.5660





class SystemsWeight(om.ExplicitComponent):
    """Computes systems weight based on Raymer's equations."""
    
    def setup(self):

        self.add_input("w_avionics", val=1.0, units="N", desc="Avionics weight")
        self.add_input("w_electrical", val=1.0, units="N", desc="Electrical systems weight")
        self.add_input("w_lg", val=1.0, units="N", desc="Landing gear weight")
        self.add_input("w_fcs", val=1.0, units="N", desc="Flight control systems weight")
        
        self.add_output("w_systems", units="N",
                       desc="Systems weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        w_avionics = inputs["w_avionics"]
        w_electrical = inputs["w_electrical"]
        w_lg = inputs["w_lg"]
        w_fcs = inputs["w_fcs"]

        outputs["w_systems"] = w_avionics + w_electrical + w_lg + w_fcs 


    def compute_partials(self, inputs, partials):
        partials["w_systems", "w_avionics"] = 1.0
        partials["w_systems", "w_electrical"] = 1.0
        partials["w_systems", "w_lg"] = 1.0
        partials["w_systems", "w_fcs"] = 1.0

class AircraftWeight(om.ExplicitComponent):
    """Computes total aircraft weight by summing component weights."""
    
    def setup(self):
        self.add_input("w_wing", val=1.0, units="N", desc="Wing weight")
        self.add_input("w_fuselage", val=1.0, units="N", desc="Fuselage weight")
        self.add_input("w_systems", val=1.0, units="N", desc="Systems weight")
        self.add_input("w_furnishings", val=1.0, units="N", desc="Furnishings weight")
        self.add_input("w_hybrid_ptrain", val=1.0, units="N", desc="Powertrain weight")
        self.add_input("w_pay", val=1.0, units="N", desc="Payload weight")
        self.add_input("w_htail", val=1.0, units="N", desc="Horizontal tail weight")
        self.add_input("w_vtail", val=1.0, units="N", desc="Vertical tail weight")
        self.add_input("w_fuel", val=1.0, units="N", desc="Fuel weight")

        self.add_output("w_airframe", units="N",
                       desc="Total structural weight")
        
        self.add_output("w_mto_calc", units="N",
                       desc="Total maximum takeoff weight")
        
        self.add_output("w_empty", units="N",
                       desc="Total empty weight")

        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        w_airframe = inputs["w_wing"] + inputs["w_fuselage"] + inputs["w_htail"] + inputs["w_vtail"]
        outputs["w_airframe"] = w_airframe
                             
        outputs["w_mto_calc"] = w_airframe + inputs["w_hybrid_ptrain"] + inputs["w_pay"] + inputs["w_systems"] + inputs["w_furnishings"] + inputs["w_fuel"]
        outputs["w_empty"] = w_airframe + inputs["w_hybrid_ptrain"] +  inputs["w_systems"] + inputs["w_furnishings"]

    def compute_partials(self, inputs, partials):

        partials["w_airframe", "w_wing"] = 1.0
        partials["w_airframe", "w_fuselage"] = 1.0
        partials["w_airframe", "w_htail"] = 1.0
        partials["w_airframe", "w_vtail"] = 1.0
        partials["w_mto_calc", "w_hybrid_ptrain"] = 1.0
        partials["w_mto_calc", "w_pay"] = 1.0
        partials["w_mto_calc", "w_systems"] = 1.0
        partials["w_mto_calc", "w_furnishings"] = 1.0
        partials["w_mto_calc", "w_fuel"] = 1.0

        partials["w_airframe", "w_wing"] = 1.0
        partials["w_airframe", "w_fuselage"] = 1.0
        partials["w_airframe", "w_htail"] = 1.0
        partials["w_airframe", "w_vtail"] = 1.0
        partials["w_empty", "w_hybrid_ptrain"] = 1.0
        partials["w_empty", "w_pay"] = 1.0
        partials["w_empty", "w_systems"] = 1.0
        partials["w_empty", "w_furnishings"] = 1.0
        partials["w_empty", "w_fuel"] = 0.0


class ComputeWeights(om.Group):
    """Group containing all weight calculation components."""
    
    def setup(self):

        self.set_input_defaults("w_mto", val=5500*9.806, units="N")


        self.add_subsystem("hybrid_ptrain_weight", HybridPowertrainWeight(), promotes_inputs=["*"], promotes_outputs=["w_hybrid_ptrain"])
        self.add_subsystem("htail_weight", HorizontalTailWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("vtail_weight", VerticalTailWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("wing_weight", WingWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("fuselage_weight", FuselageWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("landing_gear_weight", LandingGearWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("furnishings_weight", FurnishingsWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("avionics_weight", AvionicsWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("flight_control_system_weight", FlightControlSystemWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("electrical_system_weight", ElectricalSystemWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("systems_weight", SystemsWeight(), promotes_inputs=["*"], promotes_outputs=["*"])


        # Add aircraft weight calculator
        self.add_subsystem("aircraft_weight",
                          AircraftWeight(),
                          promotes_inputs=["*"],
                          promotes_outputs=["*"])
        
        #self.connect("w_mto_calc", "w_mto")
        self.connect("hybrid_ptrain_weight.compute_fuel_system_weight.w_fuel_system", "w_fuel_system")
        
        # Connect all weights to aircraft weight calculator
        #self.nonlinear_solver = om.NewtonSolver()
        #self.nonlinear_solver.options['maxiter'] = 20
        #self.nonlinear_solver.options['atol'] = 1e-6
        #self.nonlinear_solver.options['rtol'] = 1e-6
        #self.nonlinear_solver.options['solve_subsystems'] = True

        # Add linear solver
        #self.linear_solver = om.DirectSolver()
    


if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem(reports=False)
    
    # Create independent variable component
    ivc = om.IndepVarComp()

    q_cruise = 1.225 * 0.5 * 60**2
    g = 9.806
    m_fuel = 600
    w_fuel = m_fuel * g
    rho_fuel = 800
    v_fuel_m3 = m_fuel / rho_fuel
    v_fuel_gal = v_fuel_m3 * 264.172

    # WingWeight inputs
    #ivc.add_output("w_mto", val=1.0, units="lbf", desc="Maximum takeoff weight")
    ivc.add_output("n_ult", val=3.75, units=None, desc="Ultimate load factor")
    ivc.add_output("s_ref", val=282.74, units="ft**2", desc="Wing area")
    ivc.add_output("ar_w", val=14.0, units=None, desc="Wing aspect ratio")
    ivc.add_output("t_c_w", val=0.13, units=None, desc="Wing thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_w", val=10/180 * np.pi, units="rad", desc="Wing quarter-chord sweep angle")
    ivc.add_output("q_cruise", val=q_cruise, units="Pa", desc="Cruise dynamic pressure")
    ivc.add_output("lambda_w", val=0.45, units=None, desc="Wing taper ratio")
    ivc.add_output("w_fuel", val=w_fuel, units="N", desc="Weight of fuel in wings")

    # HorizontalTailWeight inputs
    # (uses n_ult, w_mto, q_cruise, ar_w from above)
    ivc.add_output("s_ht", val=30, units="ft**2", desc="Horizontal tail area")
    ivc.add_output("t_c_ht", val=0.17, units=None, desc="Horizontal tail thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_ht", val=45/180 * np.pi, units="rad", desc="Horizontal tail quarter-chord sweep angle")
    ivc.add_output("lambda_ht", val=0.45, units=None, desc="Horizontal tail taper ratio")

    # VerticalTailWeight inputs
    # (uses n_ult, w_mto, q_cruise, ar_w from above)
    ivc.add_output("f_tail", val=1.0, units=None, desc="Tail shape factor")
    ivc.add_output("s_vt", val=36, units="ft**2", desc="Vertical tail area")
    ivc.add_output("t_c_vt", val=0.17, units=None, desc="Vertical tail thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_vt", val=35/180 * np.pi, units="rad", desc="Vertical tail quarter-chord sweep angle")
    ivc.add_output("lambda_vt", val=0.45, units=None, desc="Vertical tail taper ratio")

    # AvionicsWeight inputs
    ivc.add_output("w_uav", val=100, units="lbf", desc="Uninstalled avionics weight")

    # FlightControlSystemWeight inputs
    # (uses n_ult, w_mto from above)
    ivc.add_output("l_fs", val=40, units="ft", desc="Fuselage structure length")
    ivc.add_output("b_w", val=43, units="ft", desc="Wingspan")

    # FuselageWeight inputs
    # (uses w_mto, n_ult, q_cruise, l_fs from above)
    ivc.add_output("d_fs", val=5, units="ft", desc="Fuselage structural diameter")
    ivc.add_output("l_ht", val=20, units="ft", desc="Horizontal tail lever arm length")
    ivc.add_output("v_fp", val=1e-6, units="ft**3", desc="Fuselage pressurized volume")
    ivc.add_output("delta_p", val=9, units="psi", desc="Fuselage pressurization delta")
    ivc.add_output("s_fus", val=650, units="ft**2", desc="Fuselage wetted area")

    # LandingGearWeight inputs
    # (uses w_mto, w_fuel from above)
    ivc.add_output("n_l", val=3.5, units=None, desc="Ultimate landing load factor")
    ivc.add_output("l_m", val=1, units="ft", desc="Length of main landing gear strut")
    ivc.add_output("l_n", val=1, units="ft", desc="Length of nose landing gear strut")

    # Electrical system inputs
    ivc.add_output("p_elec", val=1000000.0, units="W")
    ivc.add_output("p_bat", val=400000.0, units="W")
    ivc.add_output("e_bat", val=1000000000.0, units="J")
    ivc.add_output("p_motor", val=125000.0, units="W")
    ivc.add_output("p_gen_unit", val=300000.0, units="W")
    ivc.add_output("n_motors", val=8.0)
    ivc.add_output("n_gens", val=2.0)

    ivc.add_output("w_pay", val=6*100.0, units="lbf", desc="Payload weight")
    
    # Power/energy densities
    ivc.add_output("sp_motor", val=5000.0, units="W/kg")
    ivc.add_output("sp_pe", val=10000.0, units="W/kg")
    ivc.add_output("sp_cbl", val=20000.0, units="W/kg")
    ivc.add_output("sp_bat", val=3000.0, units="W/kg")
    ivc.add_output("se_bat", val=200.0, units="W*h/kg")
    ivc.add_output("turbine_power_density", val=2000.0, units="W/kg")
    ivc.add_output("sp_gen", val=3000.0, units="W/kg")
    
    # Fuel system inputs
    ivc.add_output("v_fuel_tot", val=v_fuel_gal, units="galUS")
    ivc.add_output("v_fuel_int", val=1e-9, units="galUS")
    ivc.add_output("n_tank", val=2.0)
    
    # Turbine inputs
    ivc.add_output("p_turbine_unit", val=1000.0, units="hp")

    # Propeller inputs
    ivc.add_output("n_prplsrs", val=2.0)
    ivc.add_output("d_prop_blades", val=2.5, units="m")
    ivc.add_output("p_prop_shaft_max", val=250000.0, units="W")
    ivc.add_output("k_prplsr", val=31.92)
    ivc.add_output("n_prop_blades", val=5.0)

    # Ducted Fan inputs
    ivc.add_output("n_fans", val=2.0)
    ivc.add_output("d_fan_blades", val=2.5, units="m")
    ivc.add_output("p_fan_shaft_max", val=250000.0, units="W")
    ivc.add_output("k_fan", val=31.92)
    ivc.add_output("n_fan_blades", val=5.0)
    

    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('weights', ComputeWeights(), promotes_inputs=["*"])

    model.connect("n_prplsrs", "compute_prop_weight.n_prplsrs")
    model.connect("d_prop_blades", "compute_prop_weight.d_blades")
    model.connect("p_prop_shaft_max", "compute_prop_weight.p_shaft")
    model.connect("k_prplsr", "compute_prop_weight.k_prplsr")
    model.connect("n_prop_blades", "compute_prop_weight.n_blades")

    model.connect("n_fans", "compute_fan_weight.n_prplsrs")
    model.connect("d_fan_blades", "compute_fan_weight.d_blades")
    model.connect("p_fan_shaft_max", "compute_fan_weight.p_shaft")
    model.connect("k_fan", "compute_fan_weight.k_prplsr")
    model.connect("n_fan_blades", "compute_fan_weight.n_blades")
    
    # Setup problem
    prob.setup()

    # Setup linear sol
    prob.model.nonlinear_solver = om.NewtonSolver()
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    prob.set_val("w_mto", 5500*9.806, units="N")
    
    # Generate N2 diagram
    # om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    g = 9.806

    # Format results
    FormatMassResults.print_tables(prob)
    FormatMassResults.plot_results(prob)