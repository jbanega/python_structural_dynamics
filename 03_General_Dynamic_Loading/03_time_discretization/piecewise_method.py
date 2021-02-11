import numpy as np

def calculateResponse(tVector, fVector, x0, v0, k, omega_n, omega_d, xi, delta_t):
    """Function to calculate dynamic response to general loading
    Inputs:
    tVector... vector of time
    fVector... vector of force
    x0... initial position
    v0... initial velocity
    k... stiffness
    omega_n... natural frequency
    omega_d... damped natural frequency
    xi... damping ratio
    delta_t... timestep
    
    Outputs:
    position, velocity"""

    nPoints = len(tVector)      # Number of data points

    # Calculating constants
    A = np.exp(-xi * omega_n * delta_t) * ((xi / (np.sqrt(1 - (xi**2)))) * np.sin(omega_d * delta_t) + np.cos(omega_d * delta_t))
    B = np.exp(-xi * omega_n * delta_t) * ((1 / omega_d * np.sin(omega_d * delta_t)))
    C = (1 / k) * (((2 * xi) / (omega_n * delta_t)) + np.exp(-xi * omega_n * delta_t) * ((((1 - 2 * (xi**2)) / (omega_d * delta_t)) -(xi / (np.sqrt(1 - (xi**2))))) * np.sin(omega_d * delta_t) - (1 + ((2*xi) / (omega_n * delta_t))) * np.cos(omega_d * delta_t)))
    D = (1 / k) * (1 - ((2*xi) / (omega_n * delta_t)) + np.exp(-xi * omega_n * delta_t) * (((2 * (xi**2) - 1) / (omega_d * delta_t))* np.sin(omega_d * delta_t) + ((2*xi) / (omega_n * delta_t)) * np.cos(omega_d * delta_t)))

    A1 = -np.exp(-xi * omega_n * delta_t) * ((omega_n / (np.sqrt(1 - (xi**2)))) * np.sin(omega_d * delta_t))
    B1 = np.exp(-xi * omega_n * delta_t) * (np.cos(omega_d * delta_t) - ((xi) / (np.sqrt(1 - (xi**2)))) * np.sin(omega_d * delta_t))
    C1 =(1 / k) * (-(1 / delta_t) + np.exp(-xi * omega_n * delta_t) * ((((omega_n) / (np.sqrt(1 - (xi**2)))) + ((xi) / (delta_t * np.sqrt(1 - (xi**2))))) * np.sin(omega_d * delta_t) + (1 / delta_t) * np.cos(omega_d * delta_t)))
    D1 =(1 / k) * ((1 / delta_t) - (np.exp(-xi * omega_n * delta_t) / delta_t) * ((xi / (np.sqrt(1 - (xi**2)))) * np.sin(omega_d * delta_t) + np.cos(omega_d * delta_t))) 

    # Initialize arrays to hold calculated position and velocity
    position = []
    velocity = []

    for n, __ in enumerate(tVector):
        # Updating force values
        if n < nPoints - 1:
            Fn = fVector[n]
            Fn_p1 = fVector[n + 1]
        else:
            Fn = fVector[n]
            Fn_p1 = 0

        # Calculating the position and velocity at the end of the time step
        current_position = (A * x0) + (B * v0) + (C * Fn) + (D * Fn_p1)
        current_velocity = (A1 * x0) + (B1 * v0) + (C1 * Fn) + (D1 * Fn_p1)

        # Storing calculated values
        position.append(current_position)
        velocity.append(current_velocity)

        # Updating initial conditions for the next iterations
        x0 = current_position
        v0 = current_velocity
    
    return position, velocity