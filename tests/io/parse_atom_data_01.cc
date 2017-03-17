
#include <iostream>
#include <fstream>
#include <sstream>
#include <deal.II/base/conditional_ostream.h>

#include <dealiiqc/qc.h>

using namespace dealii;
using namespace dealiiqc;

int main( int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure mesh"                      << std::endl
          << "  set Mesh file = "        << SOURCE_DIR
          << "/parse_atom_data_01/refined_cube.msh"           << std::endl
          << "  set Number of initial global refinements = 1" << std::endl
          << "end" << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Atom data file = "        << SOURCE_DIR
          << "/parse_atom_data_01/atom.data"                  << std::endl
          << "end" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );
      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = config.get_dimension();

      QC<3> problem( config );
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
