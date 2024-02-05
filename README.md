# Play-With-Orbits

The script is used to plot the orbit of a pulsar and its companion. It also plots
the parametric curve - acceleration vs velocity of the pulsar.
The script is useful to make plots for the desired epoch with some customizations,
like consideration of companion, precession of the orbit, animation of the orbit, etc.

Script can be used in two ways:
1) Provide the pulsar parameters as command-line arguments.
   syntax: python3 binary_pulsars.py PSR_name P0 ap e T0 Pb omega omega_dot
2) Provide the pulsar parameters in a parameter file.
   syntax: python3 binary_pulsars.py -par parfile.par
      
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
    
