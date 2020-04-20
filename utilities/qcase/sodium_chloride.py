#! /usr/bin/python3

"""
Construct and write out atom data files into tests/data folder.
"""

from ase.build  import bulk
from molecules  import molecules_from_atoms
from lammpsdata import write_atom_data

NaCl_1x1x1_atom = bulk('NaCl',
                       'rocksalt',
                       a=2.,
                       cubic=True)

NaCl_1x1x1_molecule = molecules_from_atoms(bulk('NaCl',
                                                'rocksalt',
                                                a=2.,
                                                cubic=True),
                                           molecule_ids=[1, 1, 1, 1, 1, 1, 1, 1]
                                           )

NaCl_1x1x1_atom.set_initial_charges([1, -1, 1, -1, 1, -1, 1, -1])
NaCl_1x1x1_molecule.set_initial_charges([1, -1, 1, -1, 1, -1, 1, -1])

for i in [1, 2, 4, 8]:
    NaCl_ixixi_atom = NaCl_1x1x1_atom.repeat((i, i, i))
    NaCl_ixixi_molecule = NaCl_1x1x1_molecule.repeat((i, i, i))
    write_atom_data('../../tests/data/NaCl_{0}x{0}x{0}_atom.data'.format(i),
                    NaCl_ixixi_atom,
                    atom_style='charge')
    write_atom_data('../../tests/data/NaCl_{0}x{0}x{0}_molecule.data'.format(i),
                    NaCl_ixixi_molecule,
                    atom_style='full')
