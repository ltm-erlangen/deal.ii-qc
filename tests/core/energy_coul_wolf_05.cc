
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of NaCl nano-crystal of 512 charged atoms
// interacting exclusively through Coulomb interactions using QC approach with
// full atomistic resolution.
// The blessed output is created through the LAMMPS input script included at
// the end.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run(const double &blessed_energy);
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run(
  const double &blessed_energy)
{
  QC<dim, PotentialType, atomicity>::setup_cell_energy_molecules();
  QC<dim, PotentialType, atomicity>::setup_system();
  QC<dim, PotentialType, atomicity>::setup_fe_values_objects();
  QC<dim, PotentialType, atomicity>::update_neighbor_lists();

  MPI_Barrier(QC<dim, PotentialType, atomicity>::mpi_communicator);

  Testing::SequentialFileStream write_sequentially(
    QC<dim, PotentialType, atomicity>::mpi_communicator);

  deallog << "picked up: "
          << QC<dim, PotentialType, atomicity>::cell_molecule_data
               .cell_energy_molecules.size()
          << " number of energy molecules." << std::endl;

  const double energy =
    QC<dim, PotentialType, atomicity>::template compute<false>(
      QC<dim, PotentialType, atomicity>::locally_relevant_gradient);

  QC<dim, PotentialType, atomicity>::pcout
    << "The energy computed using PairCoulWolfManager "
    << "of charged atomistic system is: " << energy << " eV." << std::endl;

  const unsigned int total_n_neighbors = dealii::Utilities::MPI::sum(
    QC<dim, PotentialType, atomicity>::neighbor_lists.size(),
    QC<dim, PotentialType, atomicity>::mpi_communicator);

  QC<dim, PotentialType, atomicity>::pcout << "Total number of neighbors "
                                           << total_n_neighbors << std::endl;

  // Accurate to 1e-9 // TODO Check unit and conversions
  AssertThrow(std::fabs(energy - blessed_energy) <
                1e7 * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
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
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 1." << std::endl
          << "    set Y center = 1." << std::endl
          << "    set Z center = 1." << std::endl
          << "    set X extent = 2." << std::endl
          << "    set Y extent = 2." << std::endl
          << "    set Z extent = 2." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 1" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Number of atom stamps = 8" << std::endl
          << "  set Maximum cutoff radius = 100" << std::endl
          << "  set Pair potential type = Coulomb Wolf" << std::endl
          << "  set Pair global coefficients = 0.4, 1.5" << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/NaCl_2x2x2_molecule.data" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = -1." << std::endl
          << "  set Cluster radius = 100" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager, 8> problem(config);
      problem.partial_run(-527.1841428070022 /*blessed energy from LAMMPS*/);
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
LAMMPS input script for energy_coul_wolf_03 test
"""
from lammps import lammps
lmp = lammps()
lmp.command("units           metal")
lmp.command("dimension       3")
lmp.command("boundary        s s s")
lmp.command("atom_style      charge")
lmp.command("read_data NaCl_2x2x2_atom.data")
lmp.command("thermo_style custom step epair evdwl ecoul elong fnorm fmax")
lmp.command("thermo_modify format 3 %20.16g")
lmp.command("thermo_modify format 7 %20.16g")
lmp.command("pair_style coul/wolf 0.4 1.5")
lmp.command("pair_coeff * *")
lmp.command("run 0")
lmp.command("variable energy equal epair")
lmp.command("variable energy_1 equal ${energy}")
lmp.command("print       \"Energy: ${energy_1}\"")

# self_energy = 0.3577238031364605*14.399645*64 = 329.6701294857547
# lammps_result = -856.854272292757
*/