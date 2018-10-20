
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
// x-------x
// |       |          x  - vertices
// |       |
// |       |          atoms are not shown.
// x-------x
//



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem(const ConfigureQC &config)
    : QC<dim, PotentialType>(config)
  {}

  void
  partial_run();
};



template <int dim, typename PotentialType>
void
Problem<dim, PotentialType>::partial_run()
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();

  // setup_system() must have prepared the inverse_mass_matrix.
  typename QC<dim, PotentialType>::vector_t &masses =
    QC<dim, PotentialType>::inverse_mass_matrix.get_vector();

  // Get masses for comparison with blessed output.
  for (typename QC<dim, PotentialType>::vector_t::BlockType::iterator entry =
         masses.block(0).begin();
       entry != masses.block(0).end();
       entry++)
    *entry = 1. / (*entry);

  masses.compress(VectorOperation::insert);

  if (dealii::Utilities::MPI::n_mpi_processes(
        QC<dim, PotentialType>::mpi_communicator) == 1)
    masses.print(std::cout);

  QC<dim, PotentialType>::pcout
    << "\n l1 norm     = " << std::setprecision(6) << masses.l1_norm()
    << "\n l2 norm     = " << std::setprecision(6) << masses.l2_norm()
    << "\n linfty norm = " << std::setprecision(6) << masses.linfty_norm()
    << std::endl;
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
          << "  set Pair specific coefficients = 0, 0, 0.877, 1.2;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 0.2" << std::endl
          << "  set Cluster weights by type = Cell" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "4 atoms" << std::endl
          << std::endl
          << "1  atom types" << std::endl
          << std::endl
          << "Masses" << std::endl
          << std::endl
          << "    1   0.7" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1 1.0 0. 0. 0." << std::endl
          << "2 2 1 1.0 1. 0. 0." << std::endl
          << "3 3 1 1.0 0. 1. 0." << std::endl
          << "4 4 1 1.0 1. 1. 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      Problem<dim, Potential::PairLJCutManager> problem(config);
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
