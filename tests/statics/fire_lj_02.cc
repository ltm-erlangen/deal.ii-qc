
// Check the convergence of FIRE minimization scheme applied to fully atomistic
// QC system with 2 atoms interacting via Lennard-Jones potential.
//
// *-------o
// |       |          o,*  - vertices
// |       |          *    - atoms
// |       |          o    - dof sites at which gradient value is zero
// o-------*
//
// 4 entries of the gradient of the total energy are zeros.



#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/core/qc.h>
#include <deal.II-qc/atom/data_out_atom_data.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/lac/solver_fire.h>


using namespace dealii;
using namespace dealiiqc;

#define WRITE_ATOM_DATA



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
  using vector_t = typename QC<dim, PotentialType>::vector_t;

public:
  Problem (const ConfigureQC &);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();
  QC<dim, PotentialType>::update_positions();

  QC<dim, PotentialType>::pcout
      << "Energy: "
      << QC<dim, PotentialType>::compute(QC<dim, PotentialType>::locally_relevant_gradient)
      << std::endl;
}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;

      if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD)==0)
        deallog.depth_console (2);

      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center =  4"                          << std::endl
          << "    set Y center =  4"                          << std::endl
          << "    set Z center =  3.5"                        << std::endl
          << "    set X extent =  8"                         << std::endl
          << "    set Y extent =  8"                         << std::endl
          << "    set Z extent =  8"                         << std::endl
          << "    set X repetitions = 1"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 3" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 1.4"              << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/8_NaCl_atom.data"           << std::endl
          << "  set Pair potential type = LJ"                 << std::endl
          << "  set Pair global coefficients = 1.35"         << std::endl
          << "  set Pair specific coefficients = "
          "0, 0, 1., .7220;"
          "0, 1, 1., .7220;"
          "1, 1, 1., .7220;"                              << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2.99"        << std::endl
          << "  set Cluster radius = 1.2"                     << std::endl
          << "  set Cluster weights by type = SamplingPoints" << std::endl
          << "end"                                            << std::endl

          << "subsection boundary_0"                          << std::endl
          << "  set Function expressions = 0., , ,"           << std::endl
          << "end"                                            << std::endl

          << "subsection boundary_1"                          << std::endl
          << "  set Function expressions = 0., , ,"           << std::endl
          << "end"                                            << std::endl

          << "subsection boundary_2"                          << std::endl
          << "  set Function expressions = , 0., ,"           << std::endl
          << "end"                                            << std::endl

          << "subsection boundary_3"                          << std::endl
          << "  set Function expressions = , 0., ,"           << std::endl
          << "end"                                            << std::endl

          << "subsection boundary_4"                          << std::endl
          << "  set Function expressions = , , 0.,"           << std::endl
          << "end"                                            << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      Problem<dim, Potential::PairLJCutManager> problem(prm_stream);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
