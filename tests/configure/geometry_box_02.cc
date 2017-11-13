
#include "../tests.h"

#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;


// Short test to check dealiiqc::parallel::shared::Triangulation.


template<int dim>
void test (const MPI_Comm    &mpi_communicator,
           const ConfigureQC &config,
           const bool         with_artificial_cells)
{
  const unsigned int
  this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

  if (this_mpi_process==0)
    {
      if (with_artificial_cells)
        std::cout << "Triangulation with artificial cells:" << std::endl;
      else
        std::cout << "Triangulation without artificial cells:" << std::endl;
    }

  dealiiqc::parallel::shared::Triangulation<dim>
  tria (mpi_communicator,
        Triangulation<dim>::limit_level_difference_at_vertices,
        with_artificial_cells ? config.get_ghost_cell_layer_thickness() : -1.);

  config.get_geometry<dim>()->create_mesh(tria);
  tria.setup_ghost_cells();

  unsigned int n_artificial_cells = 0;

  for (auto cell = tria.begin_active(); cell != tria.end(); cell++)
    if (cell->is_artificial())
      n_artificial_cells++;

  MPI_Barrier(mpi_communicator);

  Testing::SequentialFileStream sequiential_output (mpi_communicator);

  deallog << "Number of artificial cells: "
          << n_artificial_cells
          << std::endl;

  // Destructor called automatically.
  // sequiential_output.~SequentialFileStream();
}


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  set Number of initial global refinements = 2" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 0.1"         << std::endl
          << "end"                                            << std::endl

          // Add the following to suppress error throw.
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 0.05"             << std::endl
          << "  set Pair global coefficients = 0.01"          << std::endl
          << "end"                                            << std::endl

          << "#end-of-parameter-section"                      << std::endl;


      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      test<3>(MPI_COMM_WORLD, config, false);
      test<3>(MPI_COMM_WORLD, config, true);

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
