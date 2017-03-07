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
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                                numbers::invalid_unsigned_int);
      MPI_Comm mpi_communicator(MPI_COMM_WORLD);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure mesh"                      << std::endl
	  << "  set Mesh file = "        << SOURCE_DIR
	  << "/mesh_01/hex_01.msh"                            << std::endl
	  << "  set Number of initial global refinements = 1" << std::endl
	  << "end" << std::endl;

      std::istringstream prm_stream (oss.str().c_str());
      ConfigureQC config( prm_stream );
      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = config.get_dimension();
      std::ofstream out ("output", std::ofstream::trunc);

      QC<3> problem( config );
      problem.run ();
      if( Utilities::MPI::this_mpi_process(mpi_communicator) == 0 )
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
