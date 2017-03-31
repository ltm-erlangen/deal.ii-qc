/**
 * Short test to check if QC class accepts an input parameter file.
 * Writes out a mesh in eps file format
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include <dealiiqc/qc.h>

using namespace dealii;
using namespace dealiiqc;

int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc, argv, numbers::invalid_unsigned_int);
      MPI_Comm mpi_communicator(MPI_COMM_WORLD);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure mesh"                      << std::endl
          << "  set Mesh file = "        << SOURCE_DIR
          << "/../data/hex_01.msh"                            << std::endl
          << "  set Number of initial global refinements = 1" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section"
          << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );
      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = config.get_dimension();
      std::ofstream out ("output", std::ofstream::trunc);

      QC<3> problem( config );

      if ( dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 )
        problem.write_mesh(out,"msh");
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
