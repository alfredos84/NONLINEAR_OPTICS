# NONLINEAR_OPTICS

This space is devoted to share my own codes for solving differents issues in nonlinear optics.

I deal with second- and third-order nonlinear interactions. In any case, I used the well-known Split-Step Fourier Method (SSFM) to solve the fields evolution.

Second-order interactions usually occur in nonlinear crystals. They often are inside a mirror cavity, so relective properties of mirror are needed to solve the problem. The stadard way to solve this kind of problems is through the three-wave coupled equations, devidated directly from the Maxwell's equations and they can be found in any textbook.
The implementation of this problem is in Python.

Third-order interactions can ocurr in any optical material where a high-intensity pump impinges it. However, my current codes are focused on dielectric single-mode waveguides such as standard optical fibers as well as chalcogenides. The traditional way to solve this problem is using the Generalized Nonlinear Schrodinger Equation (GNLSE). This equations describes well the light propagations in waveguides having higher-order dispersion, and effect such as self-steepening and Stimulated Raman Scattering (SRS) cannot be neglected.
My code are based on a hybrid-computing, that is, parallel and sequential computations. You should have a GPU card on your PC to run these codes.
What do I need to run the code? You need a GPU card because my code is written in the computational language C-CUDA. 

All libraries are included in the package or in the CUDA DRIVER.

For questions, please do not hesitate in contact me to: alfredo.daniel.sanchez@gmail.com
