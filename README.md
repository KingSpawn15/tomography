# tomography

Create a folder named saved_matrices and place the downloaded the .npy files there.
fields_for_tomography.py is only if you have installed meep

# Call the function interpolation_loaded_fields()
# give an array of inputarray = [t, x, y, z] in the following function
# Make sure that (data constraints):
#  sqrt(y**2 + z**2) <= 100 
#  0<x<25 and 
#  0 < t < 3

ex, ey, ez = interpolation_loaded_fields(input_array)
