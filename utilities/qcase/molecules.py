#!/usr/bin/env python

"""
Molecules object to represent a collection of atoms/molecules
"""

import copy
import numpy as np

from ase.atoms import Atoms

class Molecules(Atoms):
    """ Molecules object

    The Molecules object can represent a collection of atoms/molecules,
    or a periodically repeated structure."""

    ase_objtype = 'molecules'   # For JSONability

    def __init__(self, molecule_ids=None,
                 bonds=None,
                 symbols=None,
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
        self.bonds = bonds.copy()

    def copy(self):
        """Return a copy."""
        molecules = self.__class__(cell=self.cell,
                                   bonds=self.bonds,
                                   pbc=self.pbc, info=self.info,
                                   celldisp=self._celldisp.copy())

        molecules.arrays = {}
        for name, a in self.arrays.items():
            molecules.arrays[name] = a.copy()
        molecules.constraints = copy.deepcopy(self.constraints)
        return molecules

    def __imul__(self, m, qc_mode=False):
        """In-place repeat of atoms."""
        if isinstance(m, int):
            m = (m, m, m)

        n = len(self)
        
        self = super().__imul__(m)
        bonds = self.bonds.copy()
        unit_cell_bonds = len(bonds)

        adder = np.array([[1, 0, 2, 2]])
        adder = np.tile(adder, (unit_cell_bonds,) + (1,) * (len(bonds.shape) - 1))

        bond_idx = 0
        mol_idx  = 0
        for x in range(m[0]):
            for y in range(m[1]):
                for z in range(m[1]):

                    curr_mol_idx = mol_idx + n
                    mol_counter = mol_idx // n if qc_mode else bond_idx
                    self.arrays['molecule_ids'][mol_idx:curr_mol_idx] += mol_counter
                    mol_idx = curr_mol_idx

                    if bond_idx is not 0:
                        self.bonds = np.concatenate((self.bonds,
                                                     np.add(bonds,
                                                            adder*bond_idx)),
                                                    axis=0)
                    bond_idx = bond_idx + unit_cell_bonds

        return self

    def repeat(self, rep, qc_mode=False):
        """Create new repeated atoms object.

        The *rep* argument should be a sequence of three positive
        integers like *(2,3,1)* or a single integer (*r*) equivalent
        to *(r,r,r)*."""

        molecules = self.copy()
        if qc_mode:
            molecules.arrays['molecule_ids'].fill(1)
        return molecules.__imul__(rep, qc_mode)

    __mul__ = repeat

    def qc_molecule(self):
        """Return a Molecules object for QC problem.

        The returned object has ones for molecule_ids for
        a single unit cell of atoms."""

        return self.repeat(1, qc_mode=True)

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
