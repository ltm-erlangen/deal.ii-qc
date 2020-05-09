
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Minimize the energy of a BaTiO3 molecule,
// whose atoms interact through BornCutClass2CoulWolfManager,
// using QC approach with full atomistic resolution.
// The LAMMPS script is included at the end.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run();
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run()
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->update_neighbor_lists();
  this->update_positions();

  MPI_Barrier(this->mpi_communicator);

  Testing::SequentialFileStream write_sequentially(this->mpi_communicator);

  const auto &cell_molecules = this->cell_molecule_data.cell_energy_molecules;

  const double energy =
    this->template compute<true>(this->locally_relevant_gradient);

  this->pcout << "The energy of atomistic system before minimization is: "
              << std::fixed << std::setprecision(3) << energy << " eV."
              << std::endl
              << std::endl;

  this->pcout << "Gradient before minimization:" << std::endl;
  this->locally_relevant_gradient.print(std::cout, 3, false);
  this->pcout << std::endl;

  this->pcout << "Atoms' positions before energy minimization:" << std::endl;
  for (const auto &cell_molecule : cell_molecules)
    for (const auto &atom : cell_molecule.second.atoms)
      this->pcout << atom.position << std::endl;
  this->pcout << std::endl;

  deallog << "Minimizer output:" << std::endl;
  this->minimize_energy(-1);
  this->pcout << "Gradient after minimization:" << std::endl;
  this->locally_relevant_gradient.print(std::cout, 3, false);
  this->pcout << std::endl;

  this->pcout << "Atoms' positions after energy minimization:" << std::endl;
  for (const auto &cell_molecule : cell_molecules)
    for (const auto &atom : cell_molecule.second.atoms)
      this->pcout << atom.position << std::endl;
  this->pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss
        << "set Dimension = " << dim << std::endl

        << "subsection Geometry" << std::endl
        << "  set Type = Box" << std::endl
        << "  subsection Box" << std::endl
        << "    set X center = 2." << std::endl
        << "    set Y center = 2." << std::endl
        << "    set Z center = 2." << std::endl
        << "    set X extent = 4." << std::endl
        << "    set Y extent = 4." << std::endl
        << "    set Z extent = 4." << std::endl
        << "    set X repetitions = 1" << std::endl
        << "    set Y repetitions = 1" << std::endl
        << "    set Z repetitions = 1" << std::endl
        << "  end" << std::endl
        << "  set Number of initial global refinements = 0" << std::endl
        << "end" << std::endl

        << "subsection Configure atoms" << std::endl
        << "  set Number of atom types = 6" << std::endl
        << "  set Maximum cutoff radius = 100" << std::endl
        << "  set Pair potential type = Born Class2 Coul Wolf" << std::endl
        << "  set Pair global coefficients = 0.25, 14.5, 16.0" << std::endl
        << "  set Factor coul = 0." << std::endl
        << "  set Pair specific coefficients = "
           "*, *,    0.00,	1.0000,	0.000,	0.0000,	0.000;"
           "2, 6, 7149.81,	0.3019,	0.000,	0.0000,	0.000;"
           "4, 6, 7200.27,	0.2303,	0.000,	0.0000,	0.000;"
           "6, 6, 3719.60,	0.3408,	0.000,	597.17,	0.000;"
        << std::endl
        << "  set Bond type = Class2" << std::endl
        << "  set Bond specific coefficients = "
           "1,  2,  0.000, 149.255, 0.0,   0.0000000;"
           "3,  4,  0.000, 153.070, 0.0,  20.83333333;"
           "5,  6,  0.000,  18.465, 0.0, 208.33333333;"
        << std::endl
        << "  set Atom data file = "
        << SOURCE_DIR "/../data/BaTiO3_cs_1x1x1_qcatom.data" << std::endl
        << "end" << std::endl

        << "subsection Configure QC" << std::endl
        << "  set Ghost cell layer thickness = -1." << std::endl
        << "  set Cluster radius = 100" << std::endl
        << "  set Cluster weights by type = SamplingPoints" << std::endl
        << "end" << std::endl
        << "subsection boundary_0" << std::endl
        << "  set Function expressions = 0., 0., 0., , , , , , , , , , , , , , , , , , , , , , , , , , , ,"
        << std::endl
        << "end" << std::endl
        << "subsection Minimizer settings" << std::endl
        << "  set Max steps = 2000" << std::endl
        << "  set Tolerance = 1e-10" << std::endl
        << "  set Log history   = true" << std::endl
        << "  set Log frequency = 100" << std::endl
        << "  set Log result    = true" << std::endl
        << "  set Minimizer     = FIRE" << std::endl
        << "  subsection FIRE" << std::endl
        << "    set Initial time step   = 0.0004" << std::endl
        << "    set Maximum time step   = 0.004" << std::endl
        << "    set Maximum linfty norm = 0.02" << std::endl
        << "  end" << std::endl
        << "end" << std::endl
        << "set Number of time steps = 1000" << std::endl
        << "set Time step size = 0.0004" << std::endl
        << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairBornCutClass2CoulWolfManager, 10> problem(
        config);
      problem.partial_run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}



/*
LAMMPS input script

Vishal Boddu 08.05.2017

// actual code below
#! /usr/bin/python3
"""
LAMMPS input script for energy_born_class2_coul_wolf_07 test
"""
import math
import numpy

from lammps import lammps
from qcase.data   import barium_titanate_cs_charges

cmds = [
    "units metal",
    "atom_style full",
    "boundary   m m m",
    "fix csinfo all property/atom i_CSID",
    "read_data  ../data/BaTiO3_cs_1x1x1_atom.data fix csinfo NULL CS-Info",
    "group cores type 1 3 5",
    "group shells type 2 4 6",
    "neighbor 0.5 bin",
    "neigh_modify delay 10 check yes",
    "comm_modify vel yes",
    "compute CSequ all temp/cs cores shells",
    "compute dr all displace/atom",
    "pair_style born/coul/wolf/cs 0.25 16.0 14.5",
    "pair_coeff * *  0.0000 1.0000 0.000 0.0000 0.000",
    "pair_coeff 2 6 7149.81 0.3019 0.000 0.0000 0.000",
    "pair_coeff 4 6 7200.27 0.2303 0.000 0.0000 0.000",
    "pair_coeff 6 6 3719.60 0.3408 0.000 597.17 0.000",
    "pair_modify tail no",
    "bond_style class2",
    "bond_coeff 1 0.0 149.255 0.0     0.0000000",
    "bond_coeff 2 0.0 153.070 0.0    20.83333333",
    "bond_coeff 3 0.0  18.465 0.0   208.33333333",
    "dump dump1 all custom 100 _forces id type x y z fx fy fz",
    "thermo_style custom step etotal pe ke evdwl ecoul elong ebond fnorm",
    "thermo_modify temp CSequ",
    "thermo 100",
    "neigh_modify one 4000",
    "run 0",
    "region bottom_left block INF {0} INF {0} INF {0}".format(0.1),
    "fix 1 cores setforce 0.0 0.0 0.0 region bottom_left",
    "timestep 0.0004",
    "min_style fire",
    "minimize 0.0 1.0e-5 2000 2000"
]

lmp = lammps()

for each in cmds:
    lmp.command(each)


// ITEM: TIMESTEP
// 882
// ITEM: NUMBER OF ATOMS
// 10
// ITEM: BOX BOUNDS mm mm mm
// -3.7051936253764198e-02 4.0000000000000000e+00
// -3.7051936253764330e-02 4.0000000000000000e+00
// -3.7051936253761471e-02 4.0000000000000000e+00
// ITEM: ATOMS id type x y z fx fy fz
// 1 1 0 0 0 0 0 0
// 2 2 -0.0375286 -0.0375286 -0.0375286 -1.64284e-07 -1.64284e-07 -1.64284e-07
// 3 3 1.69649 1.69649 1.69649 -2.39239e-06 -2.39239e-06 -2.39239e-06
// 4 4 1.79706 1.79706 1.79706 -1.44106e-06 -1.44106e-06 -1.44105e-06
// 5 5 -0.0366757 1.88747 1.88747 -2.45985e-06 -2.35556e-06 -2.35556e-06
// 6 6 0.115539 1.84909 1.84909 -1.63032e-06 -1.63509e-06 -1.63509e-06
// 7 5 1.88747 -0.0366757 1.88747 -2.35556e-06 -2.45985e-06 -2.35556e-06
// 8 6 1.84909 0.115539 1.84909 -1.63509e-06 -1.63032e-06 -1.63509e-06
// 9 5 1.88747 1.88747 -0.0366757 -2.35556e-06 -2.35556e-06 -2.45985e-06
// 10 6 1.84909 1.84909 0.115539 -1.63509e-06 -1.63509e-06 -1.63032e-06
*/
