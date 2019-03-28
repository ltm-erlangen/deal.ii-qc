
// Short test to check validity of PairBornCutManager class member functions
// This test compares the results from that of LAMMPS (script at the bottom).

#include <deal.II-qc/potentials/pair_born_cut.h>

#include "../tests.h"

using namespace dealiiqc;
using namespace dealii;

void
test(const double &r,
     const double &cutoff_radius,
     const double &lammps_energy,
     const double &lammps_force)
{
  std::vector<double> born_params = {0.877, 1., 1.55, 0.877, 0};

  Potential::PairBornCutManager born(cutoff_radius);

  born.declare_interactions(0,
                            1,
                            Potential::InteractionTypes::Born,
                            born_params);

  std::pair<double, double> energy_gradient_0 =
    born.energy_and_gradient(0, 1, r * r);

  AssertThrow(Testing::almost_equal(energy_gradient_0.first,
                                    lammps_energy,
                                    500),
              ExcInternalError());
  AssertThrow(Testing::almost_equal(energy_gradient_0.second,
                                    -lammps_force,
                                    500),
              ExcInternalError());

  std::pair<double, double> energy_gradient_1 =
    born.energy_and_gradient(1, 0, r * r);

  AssertThrow(Testing::almost_equal(energy_gradient_1.first,
                                    lammps_energy,
                                    500),
              ExcInternalError());
  AssertThrow(Testing::almost_equal(energy_gradient_1.second,
                                    -lammps_force,
                                    500),
              ExcInternalError());

  // std::cout << std::numeric_limits<double>::epsilon() << std::endl;
  // The test indicates that the computations of energy and forces are
  // differ by upto 1e-11 and 1e-9 respectively.
}

int
main()
{
  try
    {
      // performing tests with blessed output (from LAMMPS)
      test(0.90, 0.95, 0.0296990839348166, -9.32160551369061);
      test(1.50, 0.95, 0., 0.);
      test(1.55, 1.75, 0.813757445403767, 0.63219011124039);

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" << std::endl;
    }

  return 0;
}


/*
#!/usr/bin/env python
""" Python script to generate LAMMPS blessed output.

Use pair_lj_cut_02/atom_data as atom data for computations.
"""

F_ATOM_DATA = open("atom_data", "w")
F_ATOM_DATA.write(
"""LAMMPS Description

     2  atoms

     2  atom types

  0.0 8000.0 xlo xhi
  0.0 8000.0 ylo yhi
  0.0 8000.0 zlo zhi

Masses

      1		1.327
      2     2.000

Atoms # full

1 1 1 +1 1.2 2.09 0.8
2 2 2 -1 1.2 2.99 0.8
""")
F_ATOM_DATA.close()

from lammps import lammps

lmp=lammps()
lmp.command("units metal")
lmp.command("dimension 3")
lmp.command("boundary s s s")
lmp.command("atom_style full")
lmp.command("read_data atom_data")

lmp.command("thermo_style custom step epair evdwl ecoul elong ebond fmax")
lmp.command("thermo_modify format 3 %20.16g")
lmp.command("thermo_modify format 7 %20.16g")

# first call
lmp.command("pair_style born 0.95")
lmp.command("pair_coeff * * 0.877 1. 1.55 .877 0")
lmp.command("run 0")
lmp.command("variable energy equal epair")
lmp.command("variable force equal fmax")
lmp.command("variable energy_1 equal ${energy}")
lmp.command("variable force_1 equal ${force}")

# second call
lmp.command("set atom 1 x 1.2  y 2.09 z 0.8")
lmp.command("set atom 1 x 2.7  y 2.09 z 0.8")
lmp.command("pair_style born 0.95")
lmp.command("pair_coeff * * 0.877 1. 1.55 .877 0.")
lmp.command("run 0")
lmp.command("variable energy_2 equal ${energy}")
lmp.command("variable force_2 equal ${force}")

# third call
lmp.command("set atom 1 x 1.20  y 2.09 z 0.8")
lmp.command("set atom 2 x 2.75  y 2.09 z 0.8")
lmp.command("pair_style born 1.75")
lmp.command("pair_coeff * * 0.877 1. 1.55 .877 0.")
lmp.command("run 0")
lmp.command("variable energy_3 equal ${energy}")
lmp.command("variable force_3 equal ${force}")


lmp.command("""print       "Energy: ${energy_1} Force: ${force_1}" """)
lmp.command("""print       "Energy: ${energy_2} Force: ${force_2}" """)
lmp.command("""print       "Energy: ${energy_3} Force: ${force_3}" """)

*/