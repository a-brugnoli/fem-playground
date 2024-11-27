import firedrake as fdrk
import ufl 
import numpy as np
np.set_printoptions(threshold=np.inf)


tol = 1e-12

def facet_internal_integral(integrand):
    return (integrand('+') + integrand('-')) * fdrk.dS 


def eliminate_zero_rows_columns(matrix, threshold=1e-12):
    # Remove rows close to zero
    rows_mask = ~np.all(np.abs(matrix) < threshold, axis=1)
    matrix = matrix[rows_mask]
    
    # Remove columns close to zero
    cols_mask = ~np.all(np.abs(matrix) < threshold, axis=0)

    matrix = matrix[:, cols_mask]
    
    return matrix

def convert_form_to_array(form):
    petsc_matrix = fdrk.assemble(form).M.handle
    array = petsc_matrix.convert("dense").getDenseArray()

    return array

def round_to_significant_digits(matrix, digits=3):
    return np.array([[round(x, digits - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0 for x in row] for row in matrix])


# Create a quadrilateral mesh with 2 elements along x and 1 along y
nx, ny = 2, 1
Lx, Ly = 1.0, 1.0  

# domain = fdrk.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
domain = fdrk.UnitSquareMesh(1, 1)
normal = fdrk.FacetNormal(domain)

cell = domain.ufl_cell()


# Zero forms part
CG_element = ufl.FiniteElement("CG", cell, 1) 

facet_CG_element = CG_element[fdrk.facet]
facet_CG_space = fdrk.FunctionSpace(domain, facet_CG_element)

brokenfacet_CG_element = fdrk.BrokenElement(facet_CG_element)
brokenfacet_CG_space = fdrk.FunctionSpace(domain, brokenfacet_CG_element)


v_0 = fdrk.TestFunction(facet_CG_space)
f_0 = fdrk.TrialFunction(facet_CG_space)
f_broken_0 = fdrk.TrialFunction(brokenfacet_CG_space)

integrand_0 = v_0 * f_0
mass_form_0 = facet_internal_integral(integrand_0)

mass_0 = convert_form_to_array(mass_form_0)
mass_int_0 = eliminate_zero_rows_columns(mass_0)

integrand_broken_0 = v_0 * f_broken_0
mass_average_form_0 = facet_internal_integral(integrand_broken_0)

mass_average_0 = convert_form_to_array(mass_average_form_0)
mass_int_average_0 = eliminate_zero_rows_columns(mass_average_0)

average_0 = np.linalg.solve(mass_int_0,mass_int_average_0)
average_0[np.abs(average_0) < tol] = 0

print("Mass 0 forms on interconnection edge")
print(mass_int_0)
print("Mass times average 0 forms on interconnection edge")
print(mass_int_average_0)
print("Average 0 forms")
print(average_0)


# Two forms part
if str(cell) == "triangle":
    # RT_element = ufl.FiniteElement("RT", cell, 1, variant="integral")
    RT_element = ufl.FiniteElement("RT", cell, 1)
else:
    RT_element = ufl.FiniteElement("RTCF", cell, 1)

facet_RT_element = RT_element[fdrk.facet]        
brokenfacet_RT_element = fdrk.BrokenElement(facet_RT_element)

brokenfacet_RT_space = fdrk.FunctionSpace(domain, brokenfacet_RT_element)
facet_RT_space = fdrk.FunctionSpace(domain, facet_RT_element)

v_2 = fdrk.TestFunction(facet_RT_space)
f_2 = fdrk.TrialFunction(facet_RT_space)
f_broken_2 = fdrk.TrialFunction(brokenfacet_RT_space)

integrand_2 = fdrk.inner(v_2, normal) * fdrk.inner(f_2, normal)
mass_form_2 = facet_internal_integral(integrand_2)

mass_2 = convert_form_to_array(mass_form_2)
mass_int_2 = eliminate_zero_rows_columns(mass_2)

integrand_broken_2 = fdrk.inner(v_2, normal) * fdrk.inner(f_broken_2, normal)
mass_average_form_2 = facet_internal_integral(integrand_broken_2)

mass_average_2 = convert_form_to_array(mass_average_form_2)
mass_int_average_2 = eliminate_zero_rows_columns(mass_average_2)

average_2 = np.linalg.solve(mass_int_2, mass_int_average_2)
average_2[np.abs(average_2) < tol] = 0
print("Mass 2 forms on interconnection edge")
print(round_to_significant_digits(mass_int_2))
print("Mass times average 2 forms on interconnection edge")
print(round_to_significant_digits(mass_int_average_2))
print("Average 2 forms")
print(round_to_significant_digits(average_2))
