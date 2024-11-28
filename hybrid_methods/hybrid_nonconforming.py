import firedrake as fdrk
import ufl
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def facet_internal_integral(integrand):
    return (integrand('+') + integrand('-')) * fdrk.dS + integrand * fdrk.ds


nel_x, nel_y = 40, 40
deg = 1
domain = fdrk.UnitSquareMesh(nel_x, nel_y)
normal = fdrk.FacetNormal(domain)

x, y = fdrk.SpatialCoordinate(domain)
cell = domain.ufl_cell()

u_exact = fdrk.sin(pi*x)*fdrk.sin(pi*y)
f = - fdrk.div(fdrk.grad(u_exact))

CR_element = fdrk.FiniteElement("CR", cell, deg)
broken_CR_element = fdrk.BrokenElement(CR_element)
CR_space = fdrk.FunctionSpace(domain, broken_CR_element)

RT_element = ufl.FiniteElement("RT", cell, deg)
facet_RT_element = RT_element[fdrk.facet]        
RT_facetspace = fdrk.FunctionSpace(domain, facet_RT_element)

mixedspace = CR_space * RT_facetspace

test_CR, test_RT = fdrk.TestFunctions(mixedspace)
trial_CR, trial_RT = fdrk.TrialFunctions(mixedspace)

a_nc_form = fdrk.dot(fdrk.grad(test_CR), fdrk.grad(trial_CR))*fdrk.dx  \
        - facet_internal_integral(test_CR*fdrk.dot(trial_RT, normal)) \
        - facet_internal_integral(fdrk.dot(test_RT, normal)*trial_CR)

l_nc_form = test_CR*f*fdrk.dx

# bc = fdrk.DirichletBC(mixedspace.sub(0), 0, "on_boundary")

sol_nonconforming = fdrk.Function(mixedspace)
problem_nonconforming = fdrk.LinearVariationalProblem(a_nc_form, l_nc_form, sol_nonconforming)
solver_nonconforming =  fdrk.LinearVariationalSolver(problem_nonconforming)
            
solver_nonconforming.solve()

sol_CR, sol_RT = sol_nonconforming.subfunctions

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121, projection="3d")

fdrk.trisurf(sol_CR, axes=ax1)
ax1.set_title("Nonconforming CR solution", fontsize=16)
ax1.set_xlabel("x-axis", fontsize=12)
ax1.set_ylabel("y-axis", fontsize=12)


CG_space = fdrk.FunctionSpace(domain, "CG", deg)
test_CG = fdrk.TestFunction(CG_space)
trial_CG = fdrk.TrialFunction(CG_space)

a_cg_form = fdrk.dot(fdrk.grad(test_CG), fdrk.grad(trial_CG))*fdrk.dx 
l_cg_form = test_CG*f*fdrk.dx

sol_CG = fdrk.Function(CG_space)
bc_CG = fdrk.DirichletBC(CG_space, 0, "on_boundary")
problem_CG = fdrk.LinearVariationalProblem(a_cg_form, l_cg_form, sol_CG, bcs=bc_CG)
solver_CG =  fdrk.LinearVariationalSolver(problem_CG)
            
solver_CG.solve()


ax2 = fig.add_subplot(122, projection="3d")
fdrk.trisurf(sol_CG, axes=ax2)
ax2.set_title("Conforming CG solution", fontsize=16)
ax2.set_xlabel("x-axis", fontsize=12)
ax2.set_ylabel("y-axis", fontsize=12)

plt.tight_layout()
plt.show()

