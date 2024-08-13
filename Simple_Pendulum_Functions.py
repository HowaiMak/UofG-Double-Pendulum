#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

def Simple_Analytic(theta0,t1,s,l):
    """
    This function returns the analytical solution for the simple pendulum.
    """
    # theta0 is the starting angle of the pendulum
    # t1 is the end time for the simulation
    # s is the step size for stepping through t
    # l is the length of the string
    
    t = np.arange(0,t1+s,s)
    tn = np.shape(t)[0]
    
    theta = np.zeros(tn)
    d_theta = np.zeros(tn)
    d2_theta = np.zeros(tn)
    
    x = np.zeros(tn)
    y = np.zeros(tn)
    
    
    for i in range(0,tn):
        theta[i] = theta0*np.cos(np.sqrt(con.g/l)*t[i])
        d_theta[i] = -np.sqrt(con.g/l)*theta0*np.sin(np.sqrt(con.g/l)*t[i])
        d2_theta[i] = -(con.g/l)*theta0*np.cos(np.sqrt(con.g/l)*t[i])
       

   
    return t,theta,d_theta,d2_theta

def Simple_Numeric(theta0,t1,s,l):
    
    """
    This function returns the numerical solution for the simple pendulum.
    """
    # theta0 is the starting angle of the pendulum
    # t1 is the end time for the simulation
    # s is the step size for stepping through t
    # l is the length of the string
    
    t = np.arange(0,t1+s,s)
    tn = np.shape(t)[0]
    
    theta = np.zeros(tn)
    d_theta = np.zeros(tn)
    d2_theta = np.zeros(tn)
    
    
    theta[0] = theta0
    d2_theta[0] = -(con.g/l)*np.sin(theta0)
    
    for i in range(1,tn):
        d_theta[i] = d_theta[i-1] + d2_theta[i-1]*s
        theta[i] = theta[i-1] + d_theta[i]*s
        d2_theta[i] = -(con.g/l)*np.sin(theta[i])
        
        
   

    return t,theta,d_theta,d2_theta

def Simple_Rk4(theta0,t1,s,l):
    
    """
    This function returns the numerical solution for the simple pendulum via a 4th order Renge-Kutta.
    """
    # theta0 is the starting angle of the pendulum
    # t1 is the end time for the simulation
    # s is the step size for stepping through t
    # l is the length of the string
    
    t = np.arange(0,t1+s,s)
    tn = np.shape(t)[0]
    
    theta = np.zeros(tn)
    d_theta = np.zeros(tn)
    d2_theta = np.zeros(tn)
    
    
    
    k1theta = np.zeros(tn)
    k1d_theta = np.zeros(tn)
    k2theta = np.zeros(tn)
    k2d_theta = np.zeros(tn)
    k3theta = np.zeros(tn)
    k3d_theta = np.zeros(tn)
    k4theta = np.zeros(tn)
    k4d_theta = np.zeros(tn)
    
        
    theta[0] = theta0
    d2_theta[0] = -(con.g/l)*np.sin(theta0)
    d_theta[0] = 0
    
    k1theta[0] = theta0
    k1d_theta[0] = 0
    
    for i in range(1,tn):
        k1d_theta[i] = s*(-(con.g/l)*np.sin(theta[i-1]))
        k1theta[i] =s*d_theta[i-1]
        
        k2d_theta[i] = s*(-(con.g/l)*np.sin(theta[i-1]+k1theta[i]/2))
        k2theta[i] =s*(d_theta[i-1]+k1d_theta[i]/2)
        
        k3d_theta[i] = s*(-(con.g/l)*np.sin(theta[i-1]+k2theta[i]/2))
        k3theta[i] =s*(d_theta[i-1]+k2d_theta[i]/2)
         
        
        k4d_theta[i] = s*(-(con.g/l)*np.sin(theta[i-1]+k3theta[i]))
        k4theta[i] =s*(d_theta[i-1]+k3d_theta[i])
        
        d_theta[i] = d_theta[i-1] + (k1d_theta[i]/6) + (k2d_theta[i]/3) + (k3d_theta[i]/3) + (k4d_theta[i]/6)
        theta[i] = theta[i-1] + (k1theta[i]/6) + (k2theta[i]/3) + (k3theta[i]/3) + (k4theta[i]/6)
        
        d2_theta[i] = -(con.g/l)*np.sin(theta[i])
        
        
    return t,theta,d_theta,d2_theta
        
    
    
    
    
    
    
    
    