#!/usr/bin/env python3
"""
This module contains utility functions from earlier CSE 6040 labs.
"""

def canonicalize_tibble (X):
    """Returns a tibble in _canonical order_."""
    # Enforce Property 1:
    var_names = sorted (X.columns)
    Y = X[var_names].copy ()
    
    # Enforce Property 2:
    Y.sort_values (by=var_names, inplace=True)
    
    # Enforce Property 3:
    Y.set_index ([list (range (0, len (Y)))], inplace=True)
    
    return Y

def tibbles_are_equivalent (A, B):
    """Given two tidy tables ('tibbles'), returns True iff they are
    equivalent.
    """
    A_canonical = canonicalize_tibble (A)
    B_canonical = canonicalize_tibble (B)
    cmp = A_canonical.eq (B_canonical)
    return cmp.all ().all ()

# eof
