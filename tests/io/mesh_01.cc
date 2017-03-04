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

      std::string parameter_filename="qc.prm";
      const unsigned int dim = 3;
      std::ofstream parameter_out;
      parameter_out.open( parameter_filename.c_str(),
			  std::ofstream::trunc        );

      parameter_out
          << "set Dimension = "          << dim               << std::endl
          << "subsection Configure mesh"                      << std::endl
	  << "  set Mesh file = "        << SOURCE_DIR
	  << "/mesh_01/hex_01.msh"                            << std::endl
	  << "  set Number of initial global refinements = 1" << std::endl
	  << "end" << std::endl;
      parameter_out.close();

      QC<dim> problem( parameter_filename );
      problem.run ();
      std::ofstream out ("output", std::ofstream::trunc);
      problem.write_mesh(out,"eps");
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
