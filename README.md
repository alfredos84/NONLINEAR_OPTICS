# NONLINEAR_OPTICS

This space is devoted to share my own codes for solving differents schemes in nonlinear optics.

The exposed codes are able to solve problems involving second- and third-order nonlinear interactions using the well-known Split-Step Fourier Method (SSFM).

## **Second-order interactions**
usually are generated in nonlinear crystals. If the crystal is placed inside a mirror cavity to form an OPO or OPG, the reflective properties of the mirrors are needed.
The stadard way to solve a three-wave mixing proccess is through the three-wave coupled equations devidated directly from the Maxwell's equations (the reader can easily find the derivation from any textbook, e.g. Nonlinear Optics from Robert Boyd). 

## **Third-order interactions** 
can ocurr in any optical material where a high-intensity pump impinges it. However, my current codes are focused on dielectric single-mode waveguides such as standard optical fibers as well as chalcogenides. The traditional way to solve this problem is using the Generalized Nonlinear Schrodinger Equation (GNLSE). This equations describes well the light propagations in waveguides having higher-order dispersion, and effect such as self-steepening and Stimulated Raman Scattering (SRS) cannot be neglected.

The codes are based on a hybrid-computing, that is, parallel and sequential computations. You should have a GPU card on your PC to run these codes.
What do I need to run the code? You just need a GPU card because my code is written in the computational language C/C++/CUDA. 


All libraries are included in the package or in the CUDA DRIVER.

For questions, please do not hesitate in contact me to: alfredo.daniel.sanchez@gmail.com
