
#include <deal.II-qc/core/qc.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;


// Compute the energy of a system of two molecule each consisting of 2 atoms
// interacting exclusively through LJ interactions.
// The two molecules are far apart that the total energy is just the
// intra molecular interaction.
//
// Each molecule has two atoms with atom types 0 and 1.
// 0  0 - do not interact
// 0  1 - interact (also 1  0 - interact)
// 1  1 - do not interact
//
//    x-----x
//    |     |
//    o  o  |          o     - atoms
//    o--o--x
//
// Only one molecule is picked up as cluster molecule consequently
// for a block of gradient there should be exactly one non-zero entry.
// The non-zero entry of gradient should be twice as much of the blessed value
// in the test energy_and_gradient_lj_01 accounting for both
// the molecules.


template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &config)
    : QC<dim, PotentialType, atomicity>(config)
  {}

  void
  partial_run(const double &blessed_energy, const double &blessed_gradient);
};



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run(
  const double &blessed_energy,
  const double &blessed_gradient)
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->update_neighbor_lists();

  this->pcout << "The number of energy molecules in the system: "
              << this->cell_molecule_data.cell_energy_molecules.size()
              << std::endl;

  const double energy =
    QC<dim, PotentialType, atomicity>::template compute<true>(
      this->locally_relevant_gradient);

  this->pcout << "Energy: " << energy << " eV" << std::endl;

  if (dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
    this->locally_relevant_gradient.print(std::cout);

  AssertThrow(Testing::almost_equal(energy, blessed_energy, 50),
              ExcInternalError());

  const double gradient = this->locally_relevant_gradient.block(0)(1);

  AssertThrow(Testing::almost_equal(gradient, blessed_gradient, 50),
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
      const unsigned int dim = 2;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 3." << std::endl
          << "    set Y center = .5" << std::endl
          << "    set X extent = 6." << std::endl
          << "    set Y extent = 1." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Number of atom types = 2" << std::endl
          << "  set Maximum cutoff radius = 6" << std::endl
          << "  set Pair potential type = LJ" << std::endl
          << "  set Pair global coefficients = 2.01 " << std::endl
          << "  set Pair specific coefficients = "
          << "      1, 2, 0.877, 1.55;"
          << "      2, 2, 0.000, 1.55;"
          << "      1, 1, 0.000, 1.55;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 6.1" << std::endl
          << "  set Cluster radius = 1.9" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "4 atoms" << std::endl
          << std::endl
          << "2  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1  1.0 0.0 0. 0." << std::endl
          << "2 1 2  1.0 0.0 1. 0." << std::endl
          << "3 2 1  1.0 2.0 0. 0." << std::endl
          << "4 2 2  1.0 2.0 1. 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairLJCutManager, 2> problem(config);
      problem.partial_run(2. * 144.324376994195, 2. * 1877.831410474777
                          /*blessed values*/);
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
