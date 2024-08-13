#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

class Initial_Conditions:
    def __init__(init, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
        """
        Args: in numeric order specific for Double Pendulum
            g: gravity
            l1: length of upper pendulum in meter
            l2: length of lower pendulum in meter
            m1: attached mass on the upper pendulum in kg
            m2: attached mass on the lower pendulum in kg
            theta1: initial angular space of the upper pendulum theta1, at t = 0
            theta2: initial angular space of the lower pendulum theta2, at t = 0
            omega1: initial angular velocity of the upper pendulum theta1, at t = 0
            omega2: initial angular velocity of the lower pendulum theta2, at t = 0
        Return:
            array of the above arguments
        """
        init.arg1 = arg1
        init.arg2 = arg2
        init.arg3 = arg3
        init.arg4 = arg4
        init.arg5 = arg5
        init.arg6 = arg6
        init.arg7 = arg7
        init.arg8 = arg8
        init.arg9 = arg9

#functions for calculating angular acceleration
def alpha_1(g, l1, l2, m1, m2, th1, th2, om1, om2):
    a1 = (-g*(2*m1 + m2)*np.sin(th1) - m2*g*np.sin(th1 - 2*th2) - 2*np.sin(th1 - th2)*m2*((om2**2)*l2 + (om1**2)*l1*np.cos(th1-th2)))/(l1*(2*m1 + m2 - m2*np.cos(2*th1 - 2*th2)))
    return a1

def alpha_2(g, l1, l2, m1, m2, th1, th2, om1, om2):
    a2 = (2*np.sin(th1 - th2)*((om1**2)*l1*(m1 + m2) + g*(m1 + m2)*np.cos(th1) + (om2**2)*l2*m2*np.cos(th1 - th2)))/(l2*(2*m1 + m2 - m2*np.cos(2*th1 - 2*th2)))
    return a2

#runge kutta step forward function for a 2nd order ODE
def rk4_step(con, kfunc1, kfunc2, step):
    """
    Args:
        con: array of arguments
        kfunc1: first k-function
        kfunc2: second k-fucntion
        step: time-step
    Return:
        theta1: array of zeros ready to be store values of theta1
        theta2: array of zeros ready to be store values of theta2
        omega1: array of zeros ready to be store values of omega1
        omega2: array of zeros ready to be store values of omega2
    """
    #calculating k1 for all values
    k1omega1 = step*(kfunc1(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8, con.arg9))
    k1omega2 = step*(kfunc2(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8, con.arg9))
    
    
    #calculating k2 for all values
    k2omega1 = step*(kfunc1(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + (k1omega1/2), con.arg9 + (k1omega2/2)))
    k2omega2 = step*(kfunc2(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + (k1omega1/2), con.arg9 + (k1omega2/2)))
    
    
    #calculating k3 for all values
    k3omega1 = step*(kfunc1(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + (k2omega1/2), con.arg9 + (k2omega2/2)))
    k3omega2 = step*(kfunc2(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + (k2omega1/2), con.arg9 + (k2omega2/2)))
    
    #calculating k4 for all values
    k4omega1 = step*(kfunc1(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + k3omega1, con.arg9 + k3omega2))
    k4omega2 = step*(kfunc2(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, con.arg6, con.arg7, con.arg8 + k3omega1, con.arg9 + k3omega2))
    
    
    #runge-kutta solutions for double pendulum 
    omega1 = con.arg8 + ((k1omega1/6) + (k2omega1/3) + (k3omega1/3) + (k4omega1/6))
    omega2 = con.arg9 + ((k1omega2/6) + (k2omega2/3) + (k3omega2/3) + (k4omega2/6))
    
    
    theta1 = con.arg6 + step*omega1
    theta2 = con.arg7 + step*omega2
    
    return(theta1, theta2, omega1, omega2)

def DP_Solver(con, t, step):
    """
    Args:
        con: array of arguments from class
        t: end time for the simulation to run up to
        step: time-step to be step forward each time
    Return:
        t: array of time using time-step "step"
        theta1: initialised arrays with initial conditions ready to store in values of theta1
        theta2: initialised arrays with initial conditions ready to store in values of theta2
        omega1: initialised arrays with initial conditions ready to store in values of omega1
        omega2: initialised arrays with initial conditions ready to store in values of omega2
    """
    # Initialise the arrays to be used
    # t is an array containing each of the timepoints that we will step forward h to
    t = np.arange(0,t+(step),step)
    # n is the number of timesteps in t
    n = np.shape(t)[0]
    
    #initialising arrays which will store results
    theta1 = np.zeros(n)
    theta2 = np.zeros(n)
    omega1 = np.zeros(n)
    omega2 = np.zeros(n)
    Potential_Energy = np.zeros(n)
    Kinetic_Energy = np.zeros(n)
    Total_Energy1 = np.zeros(n)
    Total_Energy = np.zeros(n)
    
    #set initial conditions
    theta1[0] = con.arg6
    theta2[0] = con.arg7
    omega1[0] = con.arg8
    omega2[0] = con.arg9
    Potential_Energy[0] = -(con.arg4 + con.arg5)*con.arg1*con.arg2*np.cos(theta1[0]) - con.arg5*con.arg3*con.arg1*np.cos(theta2[0]) + 3*con.arg1
    Kinetic_Energy[0] = (1/2)*con.arg4*(omega1[0]**2)*(con.arg2**2)+(1/2)*con.arg5*((omega1[0]**2)*(con.arg2**2)+(omega2[0]**2)*(con.arg3**2)+2*omega1[0]*con.arg2*omega2[0]*con.arg3*np.cos(theta1[0] - theta2[0]))
    Total_Energy[0] = Potential_Energy[0] + Kinetic_Energy[0]

    #for loop repeatedly steps forward in time using the 4th order runge kutta method
    for i in range(1,n):
        """
        can.arg: in numerical order
            g: gravity
            l1: length of upper pendulum in meter
            l2: length of lower pendulum in meter
            m1: attached mass on the upper pendulum in kg
            m2: attached mass on the lower pendulum in kg
            theta1: initial angular space of the upper pendulum theta1, at t = 0
            theta2: initial angular space of the lower pendulum theta2, at t = 0
            omega1: initial angular velocity of the upper pendulum theta1, at t = 0
            omega2: initial angular velocity of the lower pendulum theta2, at t = 0
        """
        #class sets conditions for upcoming step
        c = Initial_Conditions(con.arg1, con.arg2, con.arg3, con.arg4, con.arg5, theta1[i-1], theta2[i-1], omega1[i-1], omega2[i-1])
        
        values = rk4_step(c, alpha_1, alpha_2, step)
        theta1[i] = values[0]
        theta2[i] = values[1]
        omega1[i] = values[2]
        omega2[i] = values[3]
        
        
        #energy conservation check for the system
        Potential_Energy[i] = (-(con.arg4 + con.arg5)*con.arg1*con.arg2*np.cos(theta1[i]) - con.arg5*con.arg3*con.arg1*np.cos(theta2[i])) + 3*con.arg1
        Kinetic_Energy[i] = (1/2)*con.arg4*(omega1[i]**2)*(con.arg2**2)+(1/2)*con.arg5*((omega1[i]**2)*(con.arg2**2) + (omega2[i]**2)*(con.arg3**2) + 2*omega1[i]*con.arg2*omega2[i]*con.arg3*np.cos(theta1[i] - theta2[i]))
        Total_Energy[i] = Kinetic_Energy[i] + Potential_Energy[i]
        
    
    return(t,theta1, theta2, omega1, omega2, Potential_Energy, Kinetic_Energy, Total_Energy)
    