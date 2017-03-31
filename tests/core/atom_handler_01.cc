
#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <dealiiqc/atom/atom_handler.h>
#include <dealiiqc/qc.h>
#include <dealiiqc/utility.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to check parse_atoms_and_assign_to_cells() function of AtomHandler

template<int dim>
class TestAtomHandler : public AtomHandler<dim>
{
public:

  MPI_Comm mpi_communicator;
  ConfigureQC configure_qc;
  parallel::shared::Triangulation<dim> triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  TestAtomHandler(const ConfigureQC &config)
    :
    mpi_communicator(MPI_COMM_WORLD),
    configure_qc( config),
    AtomHandler<dim>( configure_qc),
    triangulation (mpi_communicator,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    fe (FE_Q<dim>(1),dim),
    dof_handler    (triangulation)
  {}

  void run()
  {
    GridGenerator::hyper_cube( triangulation, 0., 8., true );
    triangulation.refine_global (1);
    dof_handler.distribute_dofs (fe);
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler, mpi_communicator);
    AtomHandler<dim>::update_energy_atoms();
    //AtomHandler<dim>::update_cluster_weights();
    std::ofstream f("cell_data.vtk");
    AtomHandler<dim>::write_cell_data( dof_handler, f);
    f.close();
  }

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc,
          argv,
          numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/8_NaCl_atom.data"          << std::endl
          << "end" << std::endl
          << "subsection Configure qc"                        << std::endl
          << "  set Max search radius = 2.0"                  << std::endl
          << "end" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      dealiiqc::ConfigureQC config( prm_stream );

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = config.get_dimension();

      TestAtomHandler<3> problem (config);
      problem.run();

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
