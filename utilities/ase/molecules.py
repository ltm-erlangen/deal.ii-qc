#! /usr/bin/python3

"""
Molecules object to represent a collection of atoms/molecules
"""

import numpy as np

from ase.atoms import Atoms

class Molecules(Atoms):
    """ Molecules object

    The Molecules object can represent a collection of atoms/molecules,
    or a periodically repeated structure.
    """

    ase_objtype = 'molecules'   # For JSONability

    def __init__(self, molecule_ids=None, symbols=None,
                 positions=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None):

        super().__init__(symbols=symbols,
                        positions=positions,
                        tags=tags, momenta=momenta, masses=masses,
                        magmoms=magmoms, charges=charges,
                        scaled_positions=scaled_positions,
                        cell=cell, pbc=pbc, celldisp=celldisp,
                        constraint=constraint,
                        calculator=calculator,
                        info=info, velocities=velocities)
        if molecule_ids is not None:
            self.new_array('molecule_ids', molecule_ids, int)

    def __imul__(self, m):
        """In-place repeat of atoms."""
        if isinstance(m, int):
            m = (m, m, m)

        n = len(self)
        self = super().__imul__(m)

        i0 = 0
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    self.arrays['molecule_ids'][i0:i1] += i0 // n
                    i0 = i1

        return self

def molecules_from_atoms(atoms, molecule_ids):
    """
    Construct Molecule from Atoms' object.
    """
    symbols = getattr(atoms, 'symbols', None)
    positions = getattr(atoms, 'positions', None)
    tags = getattr(atoms, 'tags', None)
    momenta = getattr(atoms, 'momenta', None)
    masses = getattr(atoms, 'masses', None)
    magmoms = getattr(atoms, 'magmoms', None)
    charges = getattr(atoms, 'charges', None)
    scaled_positions = getattr(atoms, 'scaled_positions', None)
    cell = getattr(atoms, 'cell', None)
    pbc = getattr(atoms, 'pbc', None)
    celldisp = getattr(atoms, 'celldisp', None)
    constraint = getattr(atoms, 'constraint', None)
    calculator = getattr(atoms, 'calculator', None)
    info = getattr(atoms, 'info', None)
    velocities = getattr(atoms, 'velocities', None)

    return Molecules(molecule_ids,
                        symbols=symbols,
                        positions=positions,
                        tags=tags, momenta=momenta, masses=masses,
                        magmoms=magmoms, charges=charges,
                        scaled_positions=scaled_positions,
                        cell=cell, pbc=pbc, celldisp=celldisp,
                        constraint=constraint,
                        calculator=calculator,
                        info=info, velocities=velocities)
