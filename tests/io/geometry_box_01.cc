#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_out.h>

#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/utilities.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to check Geometry::Box<dim>::create_coarse_mesh()


template<int dim>
void test (const MPI_Comm &mpi_communicator, const ConfigureQC &config)
{
  parallel::shared::Triangulation<dim> tria (mpi_communicator,
                                             Triangulation<dim>::limit_level_difference_at_vertices);

  config.get_geometry<dim>()->create_coarse_mesh(tria);

  GridOut grid_out;
  grid_out.write_vtk (tria, std::cout);
}


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
          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl;


      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      ConfigureQC config( prm_stream );

      test<3>(MPI_COMM_WORLD, config);


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
