
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II-qc/core/qc.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;



// Calculate and check the correctness of the computation of masses
// WeightsByBase::compute_inverse_masses().
//
// Derived class being used: WeightsByCell
//
// +o-------o+
// o         o          +  - vertices or (atoms with atom_stamp 0)
// |         |
// |         |
// o         o          o  - atoms with atom_stamp 1 or 2
// +o-------o+
//
// Total 4 molecules each with 3 atoms.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &config)
    : QC<dim, PotentialType, atomicity>(config)
  {}

  void
  partial_run();
};



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run()
{
  this->setup_cell_energy_molecules();
  this->setup_system();

  // setup_system() must have prepared the inverse_mass_matrix.
  auto &masses = this->inverse_mass_matrix.get_vector();

  // Get masses for comparison with blessed output.
  for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
    for (auto entry = masses.block(atom_stamp).begin();
         entry != masses.block(atom_stamp).end();
         entry++)
      *entry = 1. / (*entry);

  masses.compress(VectorOperation::insert);

  if (dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
    masses.print(std::cout);

  this->pcout << "\n l1 norm     = " << std::setprecision(6) << masses.l1_norm()
              << "\n l2 norm     = " << std::setprecision(6) << masses.l2_norm()
              << "\n linfty norm = " << std::setprecision(6)
              << masses.linfty_norm() << std::endl;
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

      deallog.depth_console(10);

      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = .5" << std::endl
          << "    set Y center = .5" << std::endl
          << "    set Z center = .5" << std::endl
          << "    set X extent = 1." << std::endl
          << "    set Y extent = 1." << std::endl
          << "    set Z extent = 1." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 2.0" << std::endl
          << "  set Pair potential type = LJ" << std::endl
          << "  set Pair global coefficients = 1.99 " << std::endl
          << "  set Pair specific coefficients = 1, 1, 0.877, 1.2;" << std::endl
          << "  set Pair specific coefficients = 1, 2, 0.877, 1.2;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 0.2" << std::endl
          << "  set Cluster weights by type = Cell" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "12 atoms" << std::endl
          << std::endl
          << "3  atom types" << std::endl
          << std::endl
          << "Masses" << std::endl
          << std::endl
          << "    1   0.7" << std::endl
          << std::endl
          << "    2   0.2" << std::endl
          << std::endl
          << "    3   0.62" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1  1  1 1.0   0.   0.  0." << std::endl
          << "2  1  2 1.0   0.   0.1 0." << std::endl
          << "3  1  3 1.0   0.1  0.  0." << std::endl
          << "4  2  1 1.0   1.   0.  0." << std::endl
          << "5  2  2 1.0   0.9  0.  0." << std::endl
          << "6  2  3 1.0   1.   0.1 0." << std::endl
          << "7  3  1 1.0   0.   1.  0." << std::endl
          << "8  3  2 1.0   0.1  1.  0." << std::endl
          << "9  3  3 1.0   0.   0.9 0." << std::endl
          << "10 4  1 1.0   1.   1.  0." << std::endl
          << "11 4  2 1.0   0.9  1.  0." << std::endl
          << "12 4  3 1.0   1.  0.9  0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      Problem<dim, Potential::PairLJCutManager, 3> problem(config);
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
