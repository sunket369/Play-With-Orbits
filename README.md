# Play-With-Orbits

The script is used to plot the orbit of a pulsar and its companion. It also plots
the parametric curve - acceleration vs velocity of the pulsar.
The script is useful to make plots for the desired epoch with some customizations,
like consideration of companion, precession of the orbit, animation of the orbit, etc.

This work is inspired by Scott Ransom's orbital fitting routine: https://github.com/scottransom/presto.git

To read and understand the physics of this, visit: https://www3.mpifr-bonn.mpg.de/staff/pfreire/orbits/index.html

--------------------------------------------------------------------

## The script can be used in two ways:
1) Provide the pulsar parameters as command-line arguments. 
   syntax: `python3 binary_pulsars.py PSR_name P0 ap e T0 Pb omega omega_dot`
2) Provide the pulsar parameters in a parameter file.
   syntax: `python3 binary_pulsars.py -par parfile.par`
      
It is recommended to use the parameter file option as it is more convenient.

--------------------------------------------------------------------
      
## You can find the usage of the script by executing the following command:
`python3 psr_orb.py -usage`
      
## Example of using the par file options:
`python3 psr_orb.py -par parfile.par -mr 0.8 -date 2021-01-01 -prec -years 20 -anim`

--------------------------------------------------------------------

## Note:
You may run the script with test_parfile.par or any other parameter file you have.
You can also find example plots and animations generated using the script in the repository.
    
