from sympy import symbols, diff

J, w = symbols('J, w')

J = w**2
print(J)

dJ_dw = diff(J, w)
print(dJ_dw)

print(dJ_dw.subs({w:3}))  # Use a dictionary for substitution