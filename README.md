Code for fitting a saturation-type model for activity vs. Rossby number (period/convective overturn time).

The code was originally designed by Stephanie Douglas to analyze Halpha activity (L_Halpha/L_bol) for her 2014 paper.  For the original version of the code used in Douglas et al. (2014), see the douglas2014 branch of her repo (https://github.com/stephtdouglas/fit-rossby).

This fork includes a number of changes:
*   modest changes to port to python 3, rather than python 2.x (ie, updating a few print statements, and explicitly converting zips to lists);

*   adjusting to accept Log inputs of the activity proxy (ie, log_activity = -3. instead of activity = 10.**(-3)).  This allows the fitting process to more easily weight outliers according to their offset in log space, treating active and less active stars more equally; 

*   rossby model and likelihood function are adjusted to explicitly include a term that accounts for underestimated errors. 

*   example fits are provided in a python notebook, to allow exploration and code inspection in a web browser.

Licensed under the MIT license; see the LICENSE.txt file for more details.
