"""
Cylinder collision detection module.

This module uses the distance3d library to detect collisions between two cylinders
in 3D space. It employs the GJK (Gilbert-Johnson-Keerthi) algorithm with Nesterov
acceleration for efficient intersection testing.

Installation:
https://github.com/AlexanderFabisch/distance3d 

References:
    - distance3d documentation: https://alexanderfabisch.github.io/distance3d/
    - GJK intersection method: https://alexanderfabisch.github.io/distance3d/_autosummary/distance3d.gjk.gjk_nesterov_accelerated_primitives_intersection.html
    - Cylinder collider: https://alexanderfabisch.github.io/distance3d/_autosummary/distance3d.colliders.Cylinder.html
"""
import numpy as np
import distance3d


def collision(x1, y1, theta1, phi1, r1, l1, x2, y2, theta2, phi2, r2, l2) -> bool:

    # Precompute trigonometric values for efficiency
    ct1, st1, cp1, sp1 = np.cos(theta1), np.sin(theta1), np.cos(phi1), np.sin(phi1)
    ct2, st2, cp2, sp2 = np.cos(theta2), np.sin(theta2), np.cos(phi2), np.sin(phi2)
    
    # Construct SE(3) transformation matrix for cylinder 1
    g1 = np.array([
                   [cp1 * st1, -sp1, cp1 * ct1, x1], 
                   [sp1 * st1, cp1, sp1 * ct1, y1],
                   [ct1, 0, -st1, 0],
                   [0, 0, 0, 1]
                ])
    
    # Construct SE(3) transformation matrix for cylinder 2
    g2 = np.array([
                   [cp2 * st2, -sp2, cp2 * ct2, x2], 
                   [sp2 * st2, cp2, sp2 * ct2, y2],
                   [ct2, 0, -st2, 0],
                   [0, 0, 0, 1]
                ])
    
    # Create cylinder collider objects with their respective transformations and dimensions
    cylinder1 = distance3d.colliders.Cylinder(g1, r1, l1)
    cylinder2 = distance3d.colliders.Cylinder(g2, r2, l2)
    
    # Uses this collision checker returns bool (True: collision)
    return distance3d.gjk.gjk_nesterov_accelerated_primitives_intersection(cylinder1, cylinder2)