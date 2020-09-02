#!/usr/bin/env python

"""
Helper functions to write out LAMMPS atom data.
"""

import os
import numpy as np

from ase.parallel import paropen
from ase.calculators.lammps import Prism, convert

from qcase.data import atomic_numbers
from qcase.data import chemical_symbols

from qcase.molecules import Molecules


def write_atom_data(fileobj, atoms, specorder=None, force_skew=False,
                    prismobj=None, velocities=False, units="metal",
                    atom_style='atomic',
                    coreshell=False):
    """ Write atomic structure data to a LAMMPS data file.

    This function is shamelessly taken from ase.io.lammpsdata.write_lammps_data
    and modified so that it will write out correct molecule IDs.
    """
    if isinstance(fileobj, str):
        f = paropen(fileobj, "w", encoding="ascii")
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    # FIXME: We should add a check here that the encoding of the file object
    #        is actually ascii once the 'encoding' attribute of IOFormat objects
    #        starts functioning in implementation (currently it doesn't do
    #         anything).

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    f.write("{0} (written by ASE) \n\n".format(os.path.basename(f.name)))

    symbols = atoms.get_chemical_symbols()
    bonds = atoms.bonds
    n_atoms = len(symbols)
    n_bonds = len(bonds)
    f.write("{0} \t atoms \n".format(n_atoms))
    f.write("{0} \t bonds \n".format(n_bonds))

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    n_bond_types = len(np.unique(bonds[:,1])) if len(bonds) else 0
    f.write("{0}  atom types\n".format(n_atom_types))
    f.write("{0}  bond types\n".format(n_bond_types))
    f.write("\n\n")

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    f.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    f.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    f.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if force_skew or p.is_skewed():
        f.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    f.write("\n\n")

    f.write("Masses \n\n")
    masses = atoms.get_masses()
    unique_types = []
    for i, mass in enumerate(masses):
        atom_type = species.index(symbols[i]) + 1
        if atom_type not in unique_types:
            unique_types.append(atom_type)
            f.write("{0:>6} {1:23.17g}\n".format(atom_type, mass))
    f.write("\n\n")

    f.write("Atoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)

    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write("{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n"
                    .format(*(i + 1, s) + tuple(r)))
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                    .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        if isinstance(atoms, Molecules):
            molecule_ids = atoms.arrays['molecule_ids']
        else:
            molecule_ids = np.ones(len(charges))  # Assign all atoms to a single molecule
        for i, (q, m, r) in enumerate(zip(charges, molecule_ids, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write("{0:>6} {1:>3} {2:>3} {3:>6} {4:23.17g} {5:23.17g} "
                    "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
        if len(bonds):
            f.write('\n')
            f.write("Bonds \n\n")
            for bond in bonds:
                f.write('\t'.join(map(str,bond)))
                f.write('\n')
        if coreshell:
            f.write('\n\n')
            f.write("CS-Info \n\n")
            for index, value in np.ndenumerate(atoms.arrays['molecule_ids']):
                f.write("{} {}\n".format(index[0]+1, value))

    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        f.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            f.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    f.flush()
    if close_file:
        f.close()
