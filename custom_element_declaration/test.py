from firedrake import *
import ufl
import matplotlib.pyplot as plt
from math import pi
from custom_lagrange import CustomLagrangeElement

# Example usage
mesh = UnitSquareMesh(10, 10)
custom_element = CustomLagrangeElement("CG", 'triangle', 1)
V = FunctionSpace(mesh, custom_element)


# Demonstrate usage
u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)
# f = - div(grad(sin(pi*x)*sin(pi*y)))
f = sin(pi*x)*cos(pi*y)

# Variational problem (e.g., Poisson equation)
a_form = inner(grad(u), grad(v))*dx 
l_form = f*v*dx
bc = DirichletBC(V, 0, "on_boundary")

sol = Function(V)

problem = LinearVariationalProblem(a_form, l_form, sol, bcs=bc)
solver =  LinearVariationalSolver(problem)
solver.solve()

trisurf(sol)
plt.show()
