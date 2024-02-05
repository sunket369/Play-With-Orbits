import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from astropy.time import Time
import time
import sys

# ------------------------------------------------------------------------------------------------------------

# Usage of the script

if len(sys.argv) > 1 and sys.argv[1] == '-usage':
    print("""
    The script is used to plot the orbit of a pulsar and its companion. It also plots
    the parametric curve - acceleration vs velocity of the pulsar.
    The script is useful to make plots for the desired epoch with some customizations,
    like consideration of companion, precession of the orbit, animation of the orbit, etc.

    Script can be used in two ways:
    1) Provide the pulsar parameters as command-line arguments.
       syntax: python3 psr_orb.py -PSR_name PSR_name -P0 P0 -ap ap -e e -T0 T0 -Pb Pb -omega omega -anim
    2) Provide the pulsar parameters in a parameter file.
       syntax: python3 psr_orb.py -par parfile.par -anim
          
    It is recommended to use the parameter file option as it is more convenient.
    
    --------------------------------------------------------------------
          
    PSR_name:  Name of the pulsar
    P0:        Period of the pulsar             [s]
    ap:        Semi-major axis                  [lt-s]
    e:         Eccentricity 
    T0:        Epoch of the periastron          [MJD]
    Pb:        Orbital period of the binary     [days]
    omega:     Longitude of periastron          [deg]
    omega_dot: Rate of change of longitude of periastron [deg/year]
    M_2:       Mass of the companion            [M_sun]
    M_tot:     Total mass of the system         [M_sun]
          
    --------------------------------------------------------------------

    Additional options:
    -usage:    Usage of the script
    -par:      Parameter file
                Example: -par parfile.par
    -mr:       Mass ratio [Mp/Mc]
                Eneble the plotting of the orbit of the companion
                Example: -mr 1.2
    -date:       Julian Date [YYYY-MM-DD]
                Date at which the orbit is to be plotted
                By default, the epoch of the pulsar is used
                Example: -date 2021-01-01
    -prec:     Precession of the orbit
                For next 10 years; can be changed with '-years' option
    -years:     Number of orbits to plot
                Example: -years 10  (by default, 10 orbits are plotted)
    -interval: Interval of orbits to be plotted in years
                Example: -interval 5  (by default, 1 year interval is used)
    -anim:     Animation of the orbit
                Animated plots are saved as .gif files
                1) Motion of pulsar and its companion in the orbit at given epoch
                2) Motion of pulsar and its companion in the parametric curve
                3) Precession of the orbit for next 10 years; can be changed with '-years' option
                4) Precession of the parametric curve for next 10 years; can be changed with '-years' option
    -lim:      Limit of the extent of the animation frame
                It is a factor multiplied to max and min extent of curves
                Example: -lim 1.2  (by default, 1.5 is used)
    -colours:  Colours in the plots for the pulsar and the companion
                Example: -colours teal steelblue (by default, 'teal' and 'steelblue' are used)
          
          
          
    Example of using above options:
    python3 psr_orb.py -par parfile.par -mr 0.8 -date 2021-01-01 -prec -years 20 -anim

    --------------------------------------------------------------------

    """)
    sys.exit(1)

# ------------------------------------------------------------------------------------------------------------

def read_parameters_from_file(filename):
    """
    Reads the parameters from the specified file and returns a dictionary of the parameters.

    """
    params = {}
    # Define the parameters you are interested in
    interesting_params = {'PSR', 'PEPOCH', 'P0', 'A1', 'E', 'T0', 'PB', 'OM', 'OMDOT', 'M2', 'MTOT'}
    
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.split()
            key_upper = key.upper()
            if key_upper in interesting_params:
                params[key_upper] = value #float(value)
    
    return params

# command-line: python3 binary_pulsars.py PSR_name P0 ap e T0 Pb omega omega_dot -mr 1.2 -date 2021-01-01 -prec -years 10 -anim
# Check if command-line arguments are provided
# Check if a parameter file is provided
if '-par' in sys.argv:
    # Read parameters from the specified file
    par_file = sys.argv[sys.argv.index('-par') + 1]
    params = read_parameters_from_file(par_file)

    # Extract individual parameters from the dictionary
    PSR_name = params['PSR']
    P_epoch = float(params['PEPOCH'])
    P0 = float(params['P0'])
    ap = float(params['A1'])
    e = float(params['E'])
    T0 = float(params['T0'])
    Pb = float(params['PB'])
    omega = float(params['OM']) * np.pi/180   # [rad]

    # Check if omega_dot is provided
    if 'OMDOT' in params:
        omega_dot = float(params['OMDOT'])    # [deg/year]
    else:
        omega_dot = 0

    # Mass ratio
    if 'MTOT' in params and 'M2' in params:
        mr = (float(params['M2']) / float(params['MTOT'])) - 1  # Mass ratio: M_p / M_c
    else:
        mr = 1.0                               # Just to proceed calculations

    if '-mr' in sys.argv:                      # Mass ratio: M_p / M_c 
        mr = float(sys.argv[sys.argv.index('-mr') + 1])
    else:
        mr = 1.0     

    # Note: -date has been read in the following part of the code

    # Number of orbits
    if '-years' in sys.argv:
        num_orbits = int(sys.argv[sys.argv.index('-years') + 1])

    # Interval of years
    if '-interval' in sys.argv:
        inter_yrs = int(sys.argv[sys.argv.index('-interval') + 1])
    else:
        inter_yrs = 1

    # Limits of animation frame
    if '-lim' in sys.argv:
        lim = float(sys.argv[sys.argv.index('-lim') + 1])
    else:
        lim = 1.5

    # Colours of the Pulsar and the Companion
    if '-colours' in sys.argv:
        psr_col = sys.argv[sys.argv.index('-colours') + 1]
        comp_col = sys.argv[sys.argv.index('-colours') + 2]
    else:
        psr_col = 'teal'
        comp_col = 'steelblue'

# If no parameter file is provided, use command-line arguments
elif len(sys.argv) > 1:
    # Command-line arguments are provided
    args = sys.argv[1:]

    # Pulsar Parameters
    if '-PSR_name' in args:
        PSR_name = args[args.index('-PSR_name') + 1]
    if '-P0' in args:
        P0 = float(args[args.index('-P0') + 1])
    if '-ap' in args:
        ap = float(args[args.index('-ap') + 1])
    if '-e' in args:
        e = float(args[args.index('-e') + 1])
    if '-T0' in args:
        T0 = float(args[args.index('-T0') + 1])
    if '-Pb' in args:
        Pb = float(args[args.index('-Pb') + 1])
    if '-omega' in args:
        omega = float(args[args.index('-omega') + 1]) * np.pi/180   # [rad]
    if '-omega_dot' in args:
        omega_dot = float(args[args.index('-omega_dot') + 1])
    else:
        omega_dot = 0
    if '-mr' in args:
        mr = float(args[args.index('-mr') + 1])
    else:
        mr = 1.0
    if '-date' in args:
        julian_date_str = args[args.index('-date') + 1]
    if '-years' in args:
        num_orbits = int(args[args.index('-num_orbits') + 1])
    else:
        num_orbits = 10
    if '-inter_yrs' in args:
        inter_yrs = int(args[args.index('-inter_yrs') + 1])
    else:
        inter_yrs = 1
    if '-lim' in args:
        lim = float(args[args.index('-lim') + 1])
    else:
        lim = 1.5
    if '-psr_col' in args:
        psr_col = args[args.index('-psr_col') + 1]
    else:
        psr_col = 'teal'
    if '-comp_col' in args:
        comp_col = args[args.index('-comp_col') + 1]
    else:
        comp_col = 'steelblue'

else:
    print("Error: Either provide command-line arguments or specify a parameter file using '-par filename.par'")
    sys.exit(1)

# ------------------------------------------------------------------------------------------------------------

# LOS velocity of the pulsar
def velocity(f, omega, e, ap, pb):
    """
        Calculates the orbital velocity in the LOS of the pulsar

            Input:  omega     - longitude of periastron
                    pb        - orbital period
                    ap        - projected semi-major axis
                    e         - eccentricity
                    f         - true anomaly

            Output: v - velocity of the pulsar    
    
    """
    V = (2*np.pi/pb) * (ap / np.sqrt(1 - e**2)) * (np.cos(omega + f) +  e*np.cos(omega))
    return V

# Acceleration of the pulsar
def accelaration(f, omega, e, ap, pb):
    """
        Calculates the orbital acceleration in the LOS of the pulsar

            Input:  omega     - longitude of periastron
                    pb        - orbital period
                    ap        - projected semi-major axis
                    e         - eccentricity
                    f         - true anomaly

            Output: a - acceleration of the pulsar    
    
    """
    A = -((2*np.pi/pb)**2) * (ap / (1 - e**2)**2) * np.sin(f + omega) * (1 + e*np.cos(f))**2
    return A

# ------------------------------------------------------------------------------------------------------------

# Date conversion [JD <-> MJD]

# Using astropy.time
# def convert_to_mjd(julian_date_str):
#     """
#     Converts the given julian date string to MJD

#             Input:  julian_date_str - Julian Date in string format
#             Output: mjd             - Modified Julian Date
#     """
#     time_object = Time(julian_date_str, format='iso', scale='utc')
#     mjd = time_object.mjd

#     return mjd

# Manual calculation
def convert_to_mjd(julian_date_str):
    """
    Converts the given julian date string to MJD

            Input:  julian_date_str - Julian Date in string format
            Output: mjd             - Modified Julian Date
    """
    # Split the input date string into components
    year, month, day = map(int, julian_date_str.split('-'))

    # Julian Date Calculation
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jd = day + ((153 * m + 2) // 5) + (365 * y) + (y // 4) - (y // 100) + (y // 400) - 32045

    # Convert Julian Date to Modified Julian Date (MJD)
    mjd = jd - 2400000.5 - 0.5

    return mjd


def convert_to_julian_date(mjd):
    """
    Converts the given MJD to Julian Date

            Input:  mjd             - Modified Julian Date
            Output: julian_date_str - Julian Date in string format
    """
    time_object = Time(mjd, format='mjd', scale='utc')
    julian_date_str = time_object.iso
    date_only = julian_date_str.split(' ')[0]

    return date_only

def add_year_to_mjd(mjd, add_year):
    """
    Adds the given number of years to the given MJD

            Input:  mjd             - Modified Julian Date
                    add_year        - Number of years to add
            Output: mjd_new         - Modified Julian Date after adding the given number of years
    """
    julian_date_parts = julian_date_str.split('-')
    year = int(julian_date_parts[0]) + add_year
    julian_date_str_new = f"{year}-{julian_date_parts[1]}-{julian_date_parts[2]}"
    mjd_new = convert_to_mjd(julian_date_str_new)

    return mjd_new

# ------------------------------------------------------------------------------------------------------------

# Solve Kepler's equations

# kepler's equation 1

def kepler_eq1(t, T0, pb, pb_dot):
    """
         Kepler's equation

            Input:  t, T0, pb, pb_dot
            Output: M - mean anomaly    
    """
    M = 2*np.pi/pb * ((t-T0) - 0.5 * pb_dot/pb * (t-T0)**2)
    return M

# --------------------------------------------------------------------

# kepler's equation 2

def kepler_eq2(E, e):
    """
            Kepler's equation

            Input:  E, e
            Output: M - mean anomaly
    """
    M = E - e*np.sin(E)
    return M

# --------------------------------------------------------------------

# kepler's equation 3

def kepler_eq3(E, e):
    """
            Kepler's equation

            Input:  E, e
            Output: f - true anomaly
    """
    f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    return f

# --------------------------------------------------------------------

# Solving for Eccentric Anomaly E --> kepler's equation 2       [Numerically]

def kepler_eq_solver(M, e, iterations):
    """
            Solves Kepler's equation numerically

            Input:  M - mean anomaly
                    e - eccentricity
                    iterations - number of iterations
            Output: E - eccentric anomaly
    """
    E = M
    accuracy = 1e-6
    iteration = 0
    while iteration < iterations:
        delta = (E - e*np.sin(E)) - M
        delta_E = delta / (1 - e*np.cos(E))

        if abs(delta_E) < accuracy:
            return E

        E1 = E - delta_E
        E = E1
        iteration += 1

# --------------------------------------------------------------------

# Equation of the orbit (ellipse)
# position vector r --> from the periastron focus to the pulsar

def vector_r(true_f, ap, e, omega):
    """
            Calculates the position vector r

            Input:  true_f, ap, e, omega
            Output: x, y   - position vector r
    """
    r = ap * (1 - e**2) / (1 + e*np.cos(true_f - omega))
    x = r * np.cos(true_f)
    y = r * np.sin(true_f)
    return x, y

# ------------------------------------------------------------------------------------------------------------

# Time array - for single orbit

data_points = 1000              # Number of data points
orbits = 5                      # Number of orbits to plot

# If Julian Date is provided as a command-line argument
if '-date' in sys.argv:
    julian_date_str = sys.argv[sys.argv.index('-date') + 1]
# If not provided, use the epoch of the pulsar
else:
    julian_date_str = convert_to_julian_date(P_epoch)              # Julian Date in string format

mjd_arg = convert_to_mjd(julian_date_str)                          # [MJD]
t_orb = np.linspace(mjd_arg, mjd_arg + orbits*Pb, data_points)     # Time array [MJD]

# Omega at the time - mjd (corresponding to the given julian date)

omega_dot_rad_per_day = omega_dot * np.pi/180 * (1/365.25)         # [rad/day]
omega_mjd = omega + omega_dot_rad_per_day * (mjd_arg - T0)         # [rad]

# ---------------------------------------------------------------------------------------------

M_orb = kepler_eq1(t_orb, T0, Pb, 0)                               # Mean anomaly [rad]

E_orb = []
for M in M_orb:
    E_orb.append(kepler_eq_solver(M, e, 100))                      # Eccentric anomaly [rad]
E_orb = np.array(E_orb)

f_orb_1 = kepler_eq3(E_orb, e)                                       # True anomaly [rad]

f_orb = f_orb_1 + omega                                              # True anomaly [rad]

# ------------------------------------------------------------------------------------------------------------

# Parameters of Companion

ap_comp = ap * mr                # Semi-major axis of the companion [lt-s]
e_comp = e                       # Eccentricity of the companion
omega_comp = omega_mjd + np.pi   # Longitude of periastron - opposite phase [rad]
f_orb_comp = f_orb + np.pi       # True anomaly of the companion [rad]

# ------------------------------------------------------------------------------------------------------------

# Velocity and Acceleration of the pulsar and the companion

c = 299792458  # speed of light  # [m/s]
psr_vel = velocity(f_orb, omega_mjd, e, ap*c, Pb*86400)/1000    # [km/s]
psr_acc = accelaration(f_orb, omega_mjd, e, ap*c, Pb*86400)     # [m/s^2]

comp_vel = velocity(f_orb_comp, omega_comp, e, ap_comp*c, Pb*86400)/1000    # [km/s]
comp_acc = accelaration(f_orb_comp, omega_comp, e, ap_comp*c, Pb*86400)     # [m/s^2]

# -------------------------------------------------------------------------------

# Array of dates

if '-years' in sys.argv:
    num_orbits = int(sys.argv[sys.argv.index('-years') + 1])
else:
    num_orbits = 10

date_array_mjd = []
for i in range(num_orbits):
    date_array_mjd.append(add_year_to_mjd(julian_date_str, i))

date_array = []
for date in date_array_mjd:
    date_array.append(convert_to_julian_date(date))

# --------------------------------------------------------------------------------

# Array of omega

def give_omegas(date_str, orbits):
    """
        Calculates the argument of periastron for the given number of orbits

            Input:  date_str - Julian Date in string format
                    orbits   - Number of orbits
            Output: omega_array - Array of argument of periastron for the given number of orbits
    """

    # Convert the input date string to MJD
    mjd = convert_to_mjd(date_str)

    add_year = np.arange(1, orbits, 1)

    # Calculate the argument of periastron at the input date
    mjd_array = [mjd]
    omega_array = [omega_mjd]
    for i in add_year:
        mjd_new = add_year_to_mjd(date_str, i)
        omega_new = omega + omega_dot_rad_per_day * (mjd_new - T0)
        mjd_array.append(mjd_new)
        omega_array.append(omega_new)

    return omega_array

omega_array = give_omegas(julian_date_str, num_orbits)

# Omegas in the range 0 to 2pi (0 to 360 deg)
for i in range(len(omega_array)):
    if omega_array[i] > 2*np.pi:
        omega_array[i] = omega_array[i] - 2*np.pi

# for companion
omega_array_comp = []
for i in range(len(omega_array)):
    omega_array_comp.append(omega_array[i] + np.pi)


# --------------------------------------------------------------------------------

# Getting true anomalies for precession

def calculate_true_anomaly(t, omega):
    """
        Calculates the true anomaly for the given time t

            Input:  t     - Time array
                    omega - Argument of periastron
            Output: f_new - True anomaly
    """
    # Finding Mean Anomaly M for a given time t
    M = kepler_eq1(t, T0, Pb, 0)

    # Finding Eccentric Anomaly E [Numerically] for a given Mean Anomaly M
    E = []
    for M_val in M:
        E.append(kepler_eq_solver(M_val, e, 100))

    E = np.array(E)

    # Finding True Anomaly f
    f = kepler_eq3(E, e)

    # Adding omega to get the true anomaly w.r.t the line of nodes
    f_new = f + omega

    return f_new

def calculate_true_anomaly_all_orbits(date_str, orbits):
    """
        Calculates the true anomaly for the given number of orbits  [for precession]

            Input:  date_str - Julian Date in string format
                    orbits   - Number of orbits
            Output: f_array  - Array of true anomaly for the given number of orbits
    """

    # Convert the input date string to MJD
    mjd = convert_to_mjd(date_str)

    add_year = np.arange(1, orbits, 1)

    # Calculate the argument of periastron at the input date
    mjd_array = [mjd]
    omega_array = [omega_mjd]
    for i in add_year:
        mjd_new = add_year_to_mjd(date_str, i)
        omega_new = omega + omega_dot_rad_per_day * (mjd_new - T0)
        mjd_array.append(mjd_new)
        omega_array.append(omega_new)

    # Time arrays for the orbits
    t_array = []
    for mjd_val in mjd_array:
        t_array.append(np.linspace(mjd_val, mjd_val + Pb, data_points))

    # Calculate the true anomaly for each orbit
    f_array = []
    for i in range(len(t_array)):
        f_array.append(calculate_true_anomaly(t_array[i], omega_array[i]))

    return f_array

# for pulsar
f_array_psr = calculate_true_anomaly_all_orbits(julian_date_str, num_orbits)

# for companion
f_array_comp = []
for i in range(num_orbits):
    f_array_comp.append(f_array_psr[i] + np.pi)

# -------------------------------------------------------------------------------

# coordinates of orbits

x_array_psr = []
y_array_psr = []
for i in range(len(f_array_psr)):
    x, y = vector_r(f_array_psr[i], ap, e, omega_array[i])
    x_array_psr.append(x)
    y_array_psr.append(y)

x_array_comp = []
y_array_comp = []
for i in range(len(f_array_comp)):
    x, y = vector_r(f_array_comp[i], ap_comp, e_comp, omega_array_comp[i])
    x_array_comp.append(x)
    y_array_comp.append(y)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Discriptive plot of the orbit with Companion
plt.figure(1)

# Define the radius and arrow length
radius = ap * 0.4    
arrow_length = ap * 0.01 

# Define the figure and axes
plt.figure(figsize=(12, 8))

# Getting orbits for the pulsar and the companion
x_orb, y_orb = vector_r(f_orb, ap, e, omega_mjd)
x_comp, y_comp = vector_r(f_orb_comp, ap_comp, e_comp, omega_comp)

alpha_array = np.linspace(1, 0.2, num_orbits)

# plotting orbits
if '-prec' in sys.argv:
    for i in range(num_orbits//inter_yrs):
        i = i * inter_yrs
        plt.plot(x_array_psr[i], y_array_psr[i], color=psr_col, alpha=alpha_array[i])
        if '-mr' in sys.argv:
            plt.plot(x_array_comp[i], y_array_comp[i], color=comp_col, alpha=alpha_array[i])

else:
    plt.plot(x_orb, y_orb, color=psr_col)
    if '-mr' in sys.argv:
        plt.plot(x_comp, y_comp, color=comp_col)

plt.scatter(0, 0, color='black', marker='+')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)       # SKY plane

# -----------------------------------------------------------------

# Making Plot more descriptive

# Periastron 
x_per = ap * (1 - e) * np.cos(omega_mjd)
y_per = ap * (1 - e) * np.sin(omega_mjd)

# Apastron
x_apa = - ap * (1 + e) * np.cos(omega_mjd)
y_apa = - ap * (1 + e) * np.sin(omega_mjd)

plt.scatter(x_per, y_per, color='black', marker='+', zorder=10)
plt.scatter(x_apa, y_apa, color='black', marker='+', zorder=10)
plt.scatter((x_per + x_apa)/2, (y_per + y_apa)/2, color='black', marker='+', zorder=10)

# Semi-major axis
plt.plot(np.linspace(x_per, x_apa, 100), np.linspace(y_per, y_apa, 100), color='red', alpha=0.2)

# Angle - Omega
arc = patches.Arc((0, 0), radius, radius, angle=0, theta1=0, theta2=omega_mjd*180/np.pi, color='red', lw=1)
plt.gca().add_patch(arc)
# Arrow to the Arc
end_x = radius/2 * np.cos(omega_mjd)
end_y = radius/2 * np.sin(omega_mjd)
arrorw = patches.FancyArrowPatch((end_x, end_y), (end_x + arrow_length*np.cos(omega_mjd), end_y + arrow_length*np.sin(omega_mjd)), color='red', mutation_scale=10)
plt.gca().add_patch(arrorw)

# -------------------------------------------------------------------

# Add text
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# Add text to the plot with the specified font properties
plt.text(x_per, y_per, 'P', ha='right', fontdict=font, va='bottom')
plt.text(x_apa, y_apa, 'A', ha='left', fontdict=font, va='top')
plt.text(0, 0, 'O', ha='right', fontdict=font, va='bottom')
plt.text(radius/1.4 * np.cos(omega_mjd/2), radius/2 * np.sin(omega_mjd/1.4), r'$\omega$', ha='left', fontdict=font, va='bottom')
plt.text((x_per + x_apa)/2, (y_per + y_apa)/2, 'C', ha='left', fontdict=font, va='bottom')
plt.text(-1.5* ap * e, 0, '$\pi$', ha='right', fontdict=font, va='bottom')

# -------------------------------------------------------------------------------

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the first Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

first_orbit = Line2D([0], [0], marker='None', color='None', label=f'For first Orbit: ')
Periastron = Line2D([0], [0], marker='None', color='None', label=r'P - Periastron')
Apastron = Line2D([0], [0], marker='None', color='None', label=r'A - Apastron')
Center = Line2D([0], [0], marker='None', color='None', label=r'C - Center of the ellipse')
Focus = Line2D([0], [0], marker='None', color='None', label=r'O - One of the foci (COM of the system)')
Omega = Line2D([0], [0], marker='None', color='None', label=f'$\omega$ - Argument of periastron ({omega_mjd*180/np.pi:.2f}$\degree$)')
POS = Line2D([0], [0], marker='None', color='None', label=r'$\pi$ - Plane of the sky')

psr_orbit = Line2D([0], [0], marker='o', color=psr_col, label=f'Orbit of {PSR_name}')
comp_orbit = Line2D([0], [0], marker='o', color=comp_col, label=r'Orbit of Companion')
mass_ratio = Line2D([0], [0], marker='None', color='None', label=f'Mass Ratio: $M_p / M_c =$ {mr:.2f}')

major_axis = Line2D([0], [0], marker='None', color='None', label=f'$A_p$ = {ap:.2f} lt-sec')
eccentricity = Line2D([0], [0], marker='None', color='None', label=f'$e$ = {e:.2f}')
orbital_period = Line2D([0], [0], marker='None', color='None', label=f'$P_b$ = {Pb:.2f} days')

omega_dot_leg = Line2D([0], [0], marker='None', color='None', label=f'$\dot{{\omega}}$ = {omega_dot:.2f}$\degree$/year')

if '-interval' in sys.argv:
    note = Line2D([0], [0], marker='None', color='None', label=f'Orbits are plotted for next {num_orbits} years\nwith an interval of {inter_yrs} years')
else:
    note = Line2D([0], [0], marker='None', color='None', label=f'Orbits are plotted for next {num_orbits} years')


plt.subplots_adjust(right=0.6)

# Add the legend with the custom entry

if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, Periastron, Apastron, Center, Focus, Omega, POS, new_line, psr_orbit, comp_orbit, mass_ratio, new_line, major_axis, eccentricity, orbital_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
    if '-prec' in sys.argv:
        legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, Periastron, Apastron, Center, Focus, Omega, POS, new_line, psr_orbit, comp_orbit, mass_ratio, new_line, major_axis, eccentricity, orbital_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, Periastron, Apastron, Center, Focus, Omega, POS, new_line, psr_orbit, new_line, major_axis, eccentricity, orbital_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
    if '-prec' in sys.argv:
        legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, Periastron, Apastron, Center, Focus, Omega, POS, new_line, psr_orbit, new_line, major_axis, eccentricity, orbital_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, Periastron, Apastron, Center, Focus, Omega, POS, new_line, psr_orbit, comp_orbit, mass_ratio, new_line, major_axis, eccentricity, orbital_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')
legend.get_texts()[0].set_color('black')
legend.get_texts()[1].set_color('black')

# -------------------------------------------------------------------------------

plt.tick_params(axis='both', which='major', labelsize=16)

# Title 
if '-mr' in sys.argv:
    plt.title(f'Orbit of {PSR_name} and its Companion', fontsize=16, pad=20)
else:
    plt.title(f'Orbit of {PSR_name}', fontsize=16, pad=20)

plt.xlabel('x [lt-sec]', fontsize=16)
plt.ylabel('y [lt-sec]', fontsize=16)
plt.gca().set_aspect('equal', adjustable='box')


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Plotting Parametric Curve - Pulsar and Companion
# LOS Orbital Accelaeration vs Velocity Plot


plt.figure(2)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()  # twin x-axis
ax3 = ax1.twinx()  # twin y-axis

# -------------------------------------------------------------------------------

# vector r for all orbits
vel_array_psr = []
acc_array_psr = []
for i in range(num_orbits):
    #v = velocity(f_array_psr[i], omega_array[i], e, ap*c, Pb*86400)/1000
    v = velocity(f_array_psr[i] - omega_array[i], omega_array[i], e, ap*c, Pb*86400)/1000
    #a = accelaration(f_array_psr[i], omega_array[i], e, ap*c, Pb*86400)
    a = accelaration(f_array_psr[i] - omega_array[i], omega_array[i], e, ap*c, Pb*86400)
    vel_array_psr.append(v)
    acc_array_psr.append(a)

vel_array_comp = []
acc_array_comp = []
for i in range(num_orbits):
    #v = velocity(f_array_comp[i], omega_array_comp[i], e_comp, ap_comp*c, Pb*86400)/1000
    v = velocity(f_array_comp[i] - omega_array_comp[i], omega_array_comp[i], e_comp, ap_comp*c, Pb*86400)/1000
    #a = accelaration(f_array_comp[i], omega_array_comp[i], e_comp, ap_comp*c, Pb*86400)
    a = accelaration(f_array_comp[i] - omega_array_comp[i], omega_array_comp[i], e_comp, ap_comp*c, Pb*86400)
    vel_array_comp.append(v)
    acc_array_comp.append(a)

# -------------------------------------------------------------------------------

alpha_array = np.linspace(1, 0.2, num_orbits)

if '-prec' in sys.argv:
    for i in range(num_orbits//inter_yrs):
        i = i * inter_yrs
        ax1.plot(vel_array_psr[i], acc_array_psr[i], color=psr_col, alpha=alpha_array[i])
        if '-mr' in sys.argv:
            ax1.plot(vel_array_comp[i], acc_array_comp[i], color=comp_col, alpha=alpha_array[i])
else:
    ax1.plot(vel_array_psr[0], acc_array_psr[0], color=psr_col)
    if '-mr' in sys.argv:
        ax1.plot(vel_array_comp[0], acc_array_comp[0], color=comp_col)

ax1.set_xlabel(f"Orbital Velocity of the Pulsar in LOS [km/s]", fontsize=16)
ax1.set_ylabel(f"Orbital Acceleration of the Pulsar in LOS [m/s$^2$]", fontsize=16)

ax1.scatter(0, 0, color='k', marker='+')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.2)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.2)

new_tick_locations_x = np.linspace(np.min(vel_array_psr[0]), np.max(vel_array_psr[0]), 4)
new_tick_locations_y = np.linspace(np.min(acc_array_psr[0]), np.max(acc_array_psr[0]), 4)

def tick_function_x(psr_vel):
    barr_period_ms = P0 * (1 + (psr_vel / c)) * 1000
    return ["%.6f" % z for z in barr_period_ms]

def tick_function_y(psr_acc):
    barr_p_dot = P0 * psr_acc / c * 1e9
    return ["%.3f" % z for z in barr_p_dot]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations_x)
ax2.set_xticklabels(tick_function_x(new_tick_locations_x))
ax2.set_xlabel(r"Observed Barycentric Period $P_{obs}$ (ms)", fontsize=14)

ax3.set_ylim(ax1.get_ylim())
ax3.set_yticks(new_tick_locations_y)
ax3.set_yticklabels(tick_function_y(new_tick_locations_y), rotation=90)
ax3.set_ylabel(r"Observed Barycentric $\dot{P}_{obs}$ [x $10^9$]", fontsize=14)

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

psr_orbit = Line2D([0], [0], marker='o', color=psr_col, label=f'Orbit of {PSR_name}')
comp_orbit = Line2D([0], [0], marker='o', color=comp_col, label=r'Orbit of Companion')

first_orbit = Line2D([0], [0], marker='None', color='None', label=f'For first Orbit: ')
eccentricity = Line2D([0], [0], marker='None', color='None', label=f'$e$ = {e:.2f}')
Omega = Line2D([0], [0], marker='None', color='None', label=f'$\omega$ = {omega_mjd*180/np.pi:.2f}$\degree$')
psr_period = Line2D([0], [0], marker='None', color='None', label=f'$P_0$ = {(P0*1000):.2f} ms')

omega_dot_leg = Line2D([0], [0], marker='None', color='None', label=f'$\dot{{\omega}}$ = {omega_dot:.2f}$\degree$/year')

if '-interval' in sys.argv:
    note = Line2D([0], [0], marker='None', color='None', label=f'Curves are plotted for next {num_orbits} years\nwith an interval of {inter_yrs} years')
else:
    note = Line2D([0], [0], marker='None', color='None', label=f'Curves are plotted for next {num_orbits} years')

plt.subplots_adjust(right=0.55)

# Add the legend with the custom entry

if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, comp_orbit, new_line, first_orbit, eccentricity, Omega, psr_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
    if '-prec' in sys.argv:
        legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, comp_orbit, new_line, first_orbit, eccentricity, Omega, psr_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, new_line, first_orbit, eccentricity, Omega, psr_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
    if '-prec' in sys.argv:
        legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, new_line, first_orbit, eccentricity, Omega, psr_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')

# Title
if '-mr' in sys.argv:
    plt.title(f'Orbital Acceleration vs Velocity for {PSR_name} and its Companion', fontsize=16, pad=20)
else:
    plt.title(f'Orbital Acceleration vs Velocity for {PSR_name}', fontsize=16, pad=20)

print("""
      Plots can be saved by clicking on the save button in the figure window.""")

# show all plots if '-anim' is not provided
if '-anim' not in sys.argv:
    plt.show()

# ------------------------------------------------------------------------------------------------------------

# dont proceed further if '-anim' is not provided
if '-anim' not in sys.argv:
    print("""
          Done!
          """)
    sys.exit(0)

if '-prec' in sys.argv:
    print(f"""
        For Animation, please check the saved GIF files in the current directory.
        1. orbital_motion_{PSR_name}.gif 
        2. parametric_motion_{PSR_name}.gif
        3. precession_of_orbit_{PSR_name}.gif
        4. precession_of_parametric_curve_{PSR_name}.gif

        ...This should take less than 2-3 minutes.
        """)
else:
    print(f"""
        For Animation, please check the saved GIF files in the current directory.
        1. orbital_motion_{PSR_name}.gif 
        2. parametric_motion_{PSR_name}.gif

        ...This should take less than 2-3 minutes.
        """)


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# A function to set limits for the animation
    
def set_limits(ax, xp_array, yp_array, xc_array=None, yc_array=None):
    """
        Set the limits for the animation

            Input:  ax       - Axes object
                    xp_array - x-coordinate array of the pulsar
                    yp_array - y-coordinate array of the pulsar
                    xc_array - x-coordinate array of the companion
                    yc_array - y-coordinate array of the companion
            Output: x_min, x_max, y_min, y_max - Limits for the animation
    """
    if '-mr' in sys.argv:
        x_max = lim * np.max([np.max(xp_array), np.max(xc_array)])
        x_min = lim * np.min([np.min(xp_array), np.min(xc_array)])
        y_max = lim * np.max([np.max(yp_array), np.max(yc_array)])
        y_min = lim * np.min([np.min(yp_array), np.min(yc_array)])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        x_max = lim * np.max([np.max(xp_array)])
        x_min = lim * np.min([np.min(xp_array)])
        y_max = lim * np.max([np.max(xp_array)])
        y_min = lim * np.min([np.min(xp_array)])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


# Animation of the motion of the pulsar and the companion in the orbit

plt.figure(3)

fig, ax = plt.subplots()

# Set the size of the figure
fig.set_figwidth(12)
fig.set_figheight(8)

# Set the limits for the animation
set_limits(ax, x_array_psr[0], y_array_psr[0], x_array_comp[0], y_array_comp[0])
# --------------------------------------------------------------------

# animation will show 3 orbits

x_psr = np.tile(x_array_psr[0], 3)
y_psr = np.tile(y_array_psr[0], 3)

if '-mr' in sys.argv:
    x_comp = np.tile(x_array_comp[0], 3)
    y_comp = np.tile(y_array_comp[0], 3)


# --------------------------------------------------------------------

psr_orbit, = ax.plot([], [], 'o', color=psr_col, markersize=5)
psr_trail, = ax.plot([], [], psr_col, alpha = 0.7)

companion_orbit, = ax.plot([], [], 'o', color=comp_col, markersize=5)
companion_trail, = ax.plot([], [], comp_col, alpha = 0.7)

trail_length = 100

# ----------------------------------------------------------------------------------

# Modify the initialization function to also initialize the companion's points and trails
def init():
    psr_orbit.set_data([], [])
    psr_trail.set_data([], [])

    if '-mr' in sys.argv:
        companion_orbit.set_data([], [])
        companion_trail.set_data([], [])

        return psr_orbit, psr_trail, companion_orbit, companion_trail
    else:
        return psr_orbit, psr_trail


skip = 10

# Modify the animation function to also update the companion's points and trails
def animate(frame):

    frame *= skip

    x = x_psr[frame]
    y = y_psr[frame]
    psr_orbit.set_data(np.array([x]), np.array([y]))
    
    # Update the trail data
    trail_x = x_psr[max(0, frame-trail_length):frame]
    trail_y = y_psr[max(0, frame-trail_length):frame]
    psr_trail.set_data(trail_x, trail_y)

    if '-mr' in sys.argv:
        # Get the x and y coordinates of the companion at the current frame
        x_c = x_comp[frame]
        y_c = y_comp[frame]
        companion_orbit.set_data(np.array([x_c]), np.array([y_c]))
        
        # Update the companion's trail data
        companion_trail_x = x_comp[max(0, frame-trail_length):frame]
        companion_trail_y = y_comp[max(0, frame-trail_length):frame]
        companion_trail.set_data(companion_trail_x, companion_trail_y)

        return psr_orbit, psr_trail, companion_orbit, companion_trail
    else:
        return psr_orbit, psr_trail

# Create the animation using FuncAnimation
anim = FuncAnimation(fig, animate, frames=len(x_psr) // skip, interval=0.1, init_func=init, blit=True)

# --------------------------------------------------------------------

# Background orbit sketch

plt.plot(x_psr, y_psr, color=psr_col, alpha=0.3)
if '-mr' in sys.argv:
    plt.plot(x_comp, y_comp, color=comp_col, alpha=0.3)

plt.scatter(0, 0, color='k', marker='+')                    # Center of mass
plt.axhline(y=0, color='k', linestyle='--', alpha=0.2)      # SKY plane

plt.scatter(x_per, y_per, color='k', marker='+', zorder=10)    # Periastron
plt.scatter(x_apa, y_apa, color='k', marker='+', zorder=10)    # Apastron
plt.scatter((x_per + x_apa)/2, (y_per + y_apa)/2, color='k', marker='+', zorder=10)   # Center

# semi-major axis
plt.plot(np.linspace(x_per, x_apa, 100), np.linspace(y_per, y_apa, 100), color='red', alpha=0.2)

# Angle - Omega
arc = patches.Arc((0, 0), radius, radius, angle=0, theta1=0, theta2=omega_mjd*180/np.pi, color='red', lw=1)
plt.gca().add_patch(arc)
# Arrow to the Arc
end_x = radius/2 * np.cos(omega_mjd)
end_y = radius/2 * np.sin(omega_mjd)
arrorw = patches.FancyArrowPatch((end_x, end_y), (end_x + arrow_length*np.cos(omega_mjd), end_y + arrow_length*np.sin(omega_mjd)), color='red', mutation_scale=10)
plt.gca().add_patch(arrorw)

# --------------------------------------------------------------------

# Add text
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# Add text to the plot with the specified font properties
plt.text(x_per, y_per, 'P', ha='right', fontdict=font, va='bottom')
plt.text(x_apa, y_apa, 'A', ha='left', fontdict=font, va='top')
plt.text(0, 0, 'O', ha='right', fontdict=font, va='bottom')
plt.text(radius/1.4 * np.cos(omega_mjd/2), radius/2 * np.sin(omega_mjd/1.4), r'$\omega$', ha='left', fontdict=font, va='bottom')
plt.text((x_per + x_apa)/2, (y_per + y_apa)/2, 'C', ha='left', fontdict=font, va='bottom')
plt.text(-1.5* ap * e, 0, '$\pi$', ha='right', fontdict=font, va='bottom')

if '-mr' in sys.argv:
    psr_text = plt.text(0.02, 0.95, 'Pulsar', color=psr_col, transform=ax.transAxes, fontsize=16)
    comp_text = plt.text(0.02, 0.90, 'Companion', color=comp_col, transform=ax.transAxes, fontsize=16)

# -------------------------------------------------------------------------------

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the first Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

Periastron = Line2D([0], [0], marker='None', color='None', label=r'P - Periastron')
Apastron = Line2D([0], [0], marker='None', color='None', label=r'A - Apastron')
Center = Line2D([0], [0], marker='None', color='None', label=r'C - Center of the ellipse')
Focus = Line2D([0], [0], marker='None', color='None', label=r'O - One of the foci (COM of the system)')
Omega = Line2D([0], [0], marker='None', color='None', label=f'$\omega$ - Argument of periastron ({omega_mjd*180/np.pi:.2f}$\degree$)')
POS = Line2D([0], [0], marker='None', color='None', label=r'$\pi$ - Plane of the sky')

#psr_orbit = Line2D([0], [0], marker='None', color='None', label=r'Orbit of Pulsar')
#comp_orbit = Line2D([0], [0], marker='None', color='None', label=r'Orbit of Companion')
mass_ratio = Line2D([0], [0], marker='None', color='None', label=f'Mass Ratio: $M_p / M_c =$ {mr:.2f}')

major_axis = Line2D([0], [0], marker='None', color='None', label=f'$A_p$ = {ap:.2f} lt-sec')
eccentricity = Line2D([0], [0], marker='None', color='None', label=f'$e$ = {e:.2f}')
orbital_period = Line2D([0], [0], marker='None', color='None', label=f'$P_b$ = {Pb:.2f} days')

plt.subplots_adjust(right=0.55)

# Add the legend with the custom entry

if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, Periastron, Apastron, Center, Focus, Omega, POS, new_line, mass_ratio, major_axis, eccentricity, orbital_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, Periastron, Apastron, Center, Focus, Omega, POS, new_line, new_line, major_axis, eccentricity, orbital_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')
legend.get_texts()[0].set_color('black')
legend.get_texts()[1].set_color('black')

# -------------------------------------------------------------------------------

plt.tick_params(axis='both', which='major', labelsize=16)

if '-mr' in sys.argv:
    plt.title(f'Orbit of the Pulsar {PSR_name} and its Companion', fontsize=16)
else:
    plt.title(f'Orbit of the Pulsar {PSR_name}', fontsize=16)

plt.xlabel('x [lt-sec]', fontsize=16)
plt.ylabel('y [lt-sec]', fontsize=16)
plt.gca().set_aspect('equal', adjustable='box')


# same ratio for both axes
plt.gca().set_aspect('equal', adjustable='box')

# Save the animation as a GIF using ImageMagickWriter
writer = PillowWriter(fps=1000)  # Adjust fps (frames per second) as needed
anim.save(f'orbital_motion_{PSR_name}.gif', writer=writer)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Animation of motion in the parametric space

plt.figure(4)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()  # twin x-axis
ax3 = ax1.twinx()  # twin y-axis


# Set the x-axis and y-axis limits
set_limits(ax1, vel_array_psr[0], acc_array_psr[0], vel_array_comp[0], acc_array_comp[0])

# --------------------------------------------------------------------

vel_psr = np.tile(vel_array_psr[0], 3)
acc_psr = np.tile(acc_array_psr[0], 3)

if '-mr' in sys.argv:
    vel_comp = np.tile(vel_array_comp[0], 3)
    acc_comp = np.tile(acc_array_comp[0], 3)

# --------------------------------------------------------------------

psr_orbit, = ax1.plot([], [], 'o', color=psr_col, markersize=5)
psr_trail, = ax1.plot([], [], psr_col, alpha = 0.7)

companion_orbit, = ax1.plot([], [], 'o', color=comp_col, markersize=5)
companion_trail, = ax1.plot([], [], comp_col, alpha = 0.7)

trail_length = 100

# --------------------------------------------------------------------

def init():
    psr_orbit.set_data([], [])
    psr_trail.set_data([], [])

    if '-mr' in sys.argv:
        companion_orbit.set_data([], [])
        companion_trail.set_data([], [])

        return psr_orbit, psr_trail, companion_orbit, companion_trail
    else:
        return psr_orbit, psr_trail
    
skip = 10

def animate(frame):

    frame *= skip

    x = vel_psr[frame]
    y = acc_psr[frame]
    psr_orbit.set_data(np.array([x]), np.array([y]))
    
    # Update the trail data
    trail_x = vel_psr[max(0, frame-trail_length):frame]
    trail_y = acc_psr[max(0, frame-trail_length):frame]
    psr_trail.set_data(trail_x, trail_y)

    if '-mr' in sys.argv:
        # Get the x and y coordinates of the companion at the current frame
        x_c = vel_comp[frame]
        y_c = acc_comp[frame]
        companion_orbit.set_data(np.array([x_c]), np.array([y_c]))
        
        # Update the companion's trail data
        companion_trail_x = vel_comp[max(0, frame-trail_length):frame]
        companion_trail_y = acc_comp[max(0, frame-trail_length):frame]
        companion_trail.set_data(companion_trail_x, companion_trail_y)

        return psr_orbit, psr_trail, companion_orbit, companion_trail
    else:
        return psr_orbit, psr_trail
    
anim = FuncAnimation(fig, animate, frames=len(vel_psr) // skip, interval=0.1, init_func=init, blit=True)

# --------------------------------------------------------------------

# Background orbit sketch

ax1.plot(vel_psr, acc_psr, color=psr_col, alpha=0.3)
if '-mr' in sys.argv:
    ax1.plot(vel_comp, acc_comp, color=comp_col, alpha=0.3)

# --------------------------------------------------------------------
    

ax1.set_xlabel(f"Orbital Velocity of the Pulsar in LOS [km/s]", fontsize=16)
ax1.set_ylabel(f"Orbital Acceleration of the Pulsar in LOS [m/s$^2$]", fontsize=16)

ax1.scatter(0, 0, color='k', marker='+')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.2)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.2)

new_tick_locations_x = np.linspace(np.min(vel_array_psr[0]), np.max(vel_array_psr[0]), 4)
new_tick_locations_y = np.linspace(np.min(acc_array_psr[0]), np.max(acc_array_psr[0]), 4)

def tick_function_x(psr_vel):
    barr_period_ms = P0 * (1 + (psr_vel / c)) * 1000
    return ["%.6f" % z for z in barr_period_ms]

def tick_function_y(psr_acc):
    barr_p_dot = P0 * psr_acc / c * 1e9
    return ["%.3f" % z for z in barr_p_dot]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations_x)
ax2.set_xticklabels(tick_function_x(new_tick_locations_x))
ax2.set_xlabel(r"Observed Barycentric Period $P_{obs}$ (ms)", fontsize=14)

ax3.set_ylim(ax1.get_ylim())
ax3.set_yticks(new_tick_locations_y)
ax3.set_yticklabels(tick_function_y(new_tick_locations_y), rotation=90)
ax3.set_ylabel(r"Observed Barycentric $\dot{P}_{obs}$ [x $10^9$]", fontsize=14)

if '-mr' in sys.argv:
    psr_text = plt.text(0.70, 0.95, 'Pulsar', color=psr_col, transform=ax.transAxes, fontsize=16)
    comp_text = plt.text(0.70, 0.90, 'Companion', color=comp_col, transform=ax.transAxes, fontsize=16)


# --------------------------------------------------------------------

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

#psr_orbit = Line2D([0], [0], marker='o', color=psr_col, label=f'Orbit of {PSR_name}')
#comp_orbit = Line2D([0], [0], marker='o', color=comp_col, label=r'Orbit of Companion')

first_orbit = Line2D([0], [0], marker='None', color='None', label=f'For first Orbit: ')
eccentricity = Line2D([0], [0], marker='None', color='None', label=f'$e$ = {e:.2f}')
Omega = Line2D([0], [0], marker='None', color='None', label=f'$\omega$ = {omega_mjd*180/np.pi:.2f}$\degree$')
psr_period = Line2D([0], [0], marker='None', color='None', label=f'$P_0$ = {(P0*1000):.2f} ms')

plt.subplots_adjust(right=0.55)

# Add the legend with the custom entry

if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, eccentricity, Omega, psr_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, eccentricity, Omega, psr_period], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')

# Title
if '-mr' in sys.argv:
    plt.title(f'Accelaration vs Velocity plot for the {PSR_name} and its Companion', fontsize=16)
else:
    plt.title(f'Accelaration vs Velocity plot for the {PSR_name}', fontsize=16)
# -------------------------------------------------------------------------------
    
# Save the animation as a GIF
writer = PillowWriter(fps=5)
anim.save(f'parametric_motion_{PSR_name}.gif', writer=writer)


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

if '-prec' not in sys.argv:
    print("""
          Done!
          """)
    plt.show()
    sys.exit(0)

# --------------------------------------

# Animation for the precession of orbit with Companion
plt.figure(5)

fig, ax = plt.subplots()

# Set the size of the figure
fig.set_figwidth(12)
fig.set_figheight(8)

# Set the limits for the animation
set_limits(ax, x_array_psr, y_array_psr, x_array_comp, y_array_comp)

# --------------------------------------------------------------------

lines = [ax.plot([], [], color=psr_col)[0] for _ in range(num_orbits)]
if '-mr' in sys.argv:
    comp_lines = [ax.plot([], [], color=comp_col)[0] for _ in range(num_orbits)]


# Create a text object
date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=16)
# Create another text object for omega
omega_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=16)

# --------------------------------------------------------------------

def init():
    for line in lines:
        line.set_data([], [])
    if '-mr' in sys.argv:
        for line in comp_lines:
            line.set_data([], [])
        return lines + comp_lines
    else:
        return lines

# plot orbits after every 'interval' years

def animate(i):
    i = inter_yrs * i
    lines[i].set_data(x_array_psr[i], y_array_psr[i])
    
    date_text.set_text('Julian Date: ' + date_array[i])
    omega_text.set_text('$\omega$: ' + str(np.round(omega_array[i]*180/np.pi, 2)) + '$\degree$')

    if '-mr' in sys.argv:
        comp_lines[i].set_data(x_array_comp[i], y_array_comp[i])
        return lines[i], comp_lines[i], date_text, omega_text
    else:
        return lines[i], date_text, omega_text

anim = FuncAnimation(fig, animate , init_func=init, frames=int(num_orbits/inter_yrs), interval=1000, blit=True)

# --------------------------------------------------------------------

# Define the font properties
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.axhline(y=0, color='k', linestyle='--', alpha=0.2)
plt.text(-1.5* ap * e, 0, '$\pi$', ha='right', fontdict=font, va='bottom')

plt.scatter(0, 0, color='k', marker='+')
plt.text(0, 0, 'O', ha='right', fontdict=font, va='bottom')

# --------------------------------------------------------------------

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the first Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

psr_orbit = Line2D([0], [0], marker='None', color=psr_col, label=f'Orbit of {PSR_name}')
comp_orbit = Line2D([0], [0], marker='None', color=comp_col, label=r'Orbit of Companion')

omega_0 = Line2D([0], [0], marker='None', color='None', label=f'$\omega_0$ = {omega*180/np.pi:.2f}$\degree$')
omega_dot_leg = Line2D([0], [0], marker='None', color='None', label=f'$\dot{{\omega}}$ = {omega_dot:.2f}$\degree$ / year')
P_b = Line2D([0], [0], marker='None', color='None', label=f'Orbital Period: $P_b$ = {Pb:.2f} days') 

POS = Line2D([0], [0], marker='None', color='None', label=r'$\pi$ - Plane of the sky')
Focus = Line2D([0], [0], marker='None', color='None', label=r'O - One of the foci (COM of the system)')

if '-interval' in sys.argv:
    note = Line2D([0], [0], marker='None', color='None', label=f'Orbits are plotted for next {num_orbits} years\nwith an interval of {inter_yrs} years')
else:
    note = Line2D([0], [0], marker='None', color='None', label=f'Orbits are plotted for next {num_orbits} years')

# Adjust the plot area to make room for the legend
plt.subplots_adjust(right=0.55)

# Add the legend with the custom entry
if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, comp_orbit, new_line, omega_0, omega_dot_leg, P_b, new_line, POS, Focus, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, psr_orbit, new_line, omega_0, omega_dot_leg, P_b, new_line, POS, Focus, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')
legend.get_texts()[0].set_color('black')
legend.get_texts()[1].set_color('black')

# -------------------------------------------------------------------------------

# title
if '-mr' in sys.argv:
    plt.title(f'Precession of Orbit of {PSR_name} and its Companion', fontsize=16, pad=20)
else:
    plt.title(f'Precession of Orbit of {PSR_name}', fontsize=16, pad=20)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('x [lt-sec]', fontsize=16)
plt.ylabel('y [lt-sec]', fontsize=16)

# same ratio for both axes
plt.gca().set_aspect('equal', adjustable='box')

# -------------------------------------------------------------------------------

# Save the animation as a GIF
writer = PillowWriter(fps=5)  # Adjust fps (frames per second) as needed
anim.save(f'precession_of_orbit_{PSR_name}.gif', writer=writer)

#plt.show()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Animation for the precession of parametric curve

plt.figure(6)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()  # twin x-axis
ax3 = ax1.twinx()  # twin y-axis

# Set the x-axis and y-axis limits
set_limits(ax1, vel_array_psr, acc_array_psr, vel_array_comp, acc_array_comp)

# --------------------------------------------------------------------

lines = [ax1.plot([], [], color=psr_col)[0] for _ in range(num_orbits)]
if '-mr' in sys.argv:
    comp_lines = [ax1.plot([], [], color=comp_col)[0] for _ in range(num_orbits)]

# Create a text object
date_text = ax1.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=16)
# Create another text object for omega
omega_text = ax1.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=16)

# --------------------------------------------------------------------

def init():
    for line in lines:
        line.set_data([], [])
    if '-mr' in sys.argv:
        for line in comp_lines:
            line.set_data([], [])
        return lines + comp_lines
    else:
        return lines
    
def animate(i):

    i = i * inter_yrs

    lines[i].set_data(vel_array_psr[i], acc_array_psr[i])
    
    date_text.set_text('Date: ' + date_array[i])
    omega_text.set_text('$\omega$: ' + str(np.round(omega_array[i]*180/np.pi, 2)) + '$\degree$')

    if '-mr' in sys.argv:
        comp_lines[i].set_data(vel_array_comp[i], acc_array_comp[i])
        return lines[i], comp_lines[i], date_text, omega_text
    else:
        return lines[i], date_text, omega_text
    
anim = FuncAnimation(fig, animate, init_func=init, frames=int(num_orbits/inter_yrs), interval=1000, blit=True)

# --------------------------------------------------------------------

ax1.set_xlabel(f"Orbital Velocity of the Pulsar in LOS [km/s]", fontsize=16)
ax1.set_ylabel(f"Orbital Acceleration of the Pulsar in LOS [m/s$^2$]", fontsize=16)

ax1.scatter(0, 0, color='k', marker='+')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.2)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.2)

new_tick_locations_x = np.linspace(np.min(vel_array_psr[0]), np.max(vel_array_psr[0]), 4)
new_tick_locations_y = np.linspace(np.min(acc_array_psr[0]), np.max(acc_array_psr[0]), 4)

def tick_function_x(psr_vel):
    barr_period_ms = P0 * (1 + (psr_vel / c)) * 1000
    return ["%.6f" % z for z in barr_period_ms]

def tick_function_y(psr_acc):
    barr_p_dot = P0 * psr_acc / c * 1e9
    return ["%.3f" % z for z in barr_p_dot]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations_x)
ax2.set_xticklabels(tick_function_x(new_tick_locations_x))
ax2.set_xlabel(r"Observed Barycentric Period $P_{obs}$ (ms)", fontsize=14)

ax3.set_ylim(ax1.get_ylim())
ax3.set_yticks(new_tick_locations_y)
ax3.set_yticklabels(tick_function_y(new_tick_locations_y), rotation=90)
ax3.set_ylabel(r"Observed Barycentric $\dot{P}_{obs}$ [x $10^9$]", fontsize=14)

if '-mr' in sys.argv:
    psr_text = plt.text(0.70, 0.95, 'Pulsar', color=psr_col, transform=ax.transAxes, fontsize=16)
    comp_text = plt.text(0.70, 0.90, 'Companion', color=comp_col, transform=ax.transAxes, fontsize=16)


# --------------------------------------------------------------------

# Create a custom legend entry

# Define the font properties
font = FontProperties()
font.set_size('x-large')
font.set_style('italic')

orbit_epoch = Line2D([0], [0], marker='None', color='None', label=f'Epoch of the Orbit: ')
epoch_date = Line2D([0], [0], marker='None', color='None', label=f'{julian_date_str} (MJD {mjd_arg})' )
new_line = Line2D([0], [0], marker='None', color='None', label='')

first_orbit = Line2D([0], [0], marker='None', color='None', label=f'For first Orbit: ')
eccentricity = Line2D([0], [0], marker='None', color='None', label=f'$e$ = {e:.2f}')
Omega = Line2D([0], [0], marker='None', color='None', label=f'$\omega$ = {omega_mjd*180/np.pi:.2f}$\degree$')
psr_period = Line2D([0], [0], marker='None', color='None', label=f'$P_0$ = {(P0*1000):.2f} ms')

omega_dot_leg = Line2D([0], [0], marker='None', color='None', label=f'$\dot{{\omega}}$ = {omega_dot:.2f}$\degree$/year')

if '-interval' in sys.argv:
    note = Line2D([0], [0], marker='None', color='None', label=f'Curves are plotted for next {num_orbits} years\nwith an interval of {inter_yrs} years')
else:
    note = Line2D([0], [0], marker='None', color='None', label=f'Curves are plotted for next {num_orbits} years')

plt.subplots_adjust(right=0.55)

# Add the legend with the custom entry

if '-mr' in sys.argv:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, eccentricity, Omega, psr_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)
else:
    legend = plt.legend(handles=[orbit_epoch, epoch_date, new_line, first_orbit, eccentricity, Omega, psr_period, omega_dot_leg, new_line, note], loc='upper left', bbox_to_anchor=(1.05, 1), prop=font)

# Change the legend box color to black
legend.get_frame().set_facecolor('powderblue')
legend.get_frame().set_edgecolor('black')

# Title
if '-mr' in sys.argv:
    plt.title(f'Precession of Parametric Curve for {PSR_name} and its Companion', fontsize=16, pad=20)
else:
    plt.title(f'Precession of Parametric Curve for {PSR_name}', fontsize=16, pad=20)

# -------------------------------------------------------------------------------
    
# Save the animation as a GIF
writer = PillowWriter(fps=5)
anim.save(f'precession_of_parametric_curve_{PSR_name}.gif', writer=writer)

plt.show()

print("""
      Done!
      """)

# ------------------------------------------------------------------------------------------------------------
