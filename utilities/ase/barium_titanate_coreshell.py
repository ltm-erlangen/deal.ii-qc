#! /usr/bin/pythons3

"""
Script to generate barium titanate coreshell atom data
for LAMMPS and QC simulations.
The file names ending with qcatom.data are for QC simultions.
"""

import numpy as np

from data import barium_titanate_cs_bonds
from data import barium_titanate_cs_charges
from data import barium_titanate_cs_masses
from data import barium_titanate_cs_positions

from molecules  import Molecules
from lammpsdata import write_atom_data

born_class2_coul_wolf_02_lc = 4.00

molecule_ids = np.array([
    1, 1, # Ba core and shell
    2, 2, # Ti core and shell
    3, 3, # O  core and shell
    4, 4, # O  core and shell
    5, 5  # O  core and shell
])

positions = barium_titanate_cs_positions * born_class2_coul_wolf_02_lc

cell = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) * born_class2_coul_wolf_02_lc

BaTiO3_cs_1x1x1 = Molecules(molecule_ids=molecule_ids,
                            bonds=barium_titanate_cs_bonds,
                            symbols="BaBsTiTsOOsOOsOOs",
                            cell=cell,
                            masses=barium_titanate_cs_masses,
                            charges=barium_titanate_cs_charges,
                            positions=positions)

BaTiO3_cs_1x1x1_qc = BaTiO3_cs_1x1x1.repeat((1, 1, 1), qc_mode=True)

write_atom_data('../../tests/data/BaTiO3_cs_1x1x1_atom.data',
                BaTiO3_cs_1x1x1,
                specorder=['Ba', 'Bs', 'Ti', 'Ts', 'O', 'Os'],
                atom_style='full')
write_atom_data('../../tests/data/BaTiO3_cs_1x1x1_qcatom.data',
                BaTiO3_cs_1x1x1_qc,
                specorder=['Ba', 'Bs', 'Ti', 'Ts', 'O', 'Os'],
                atom_style='full')

if __name__ == "__main__":
    pass
