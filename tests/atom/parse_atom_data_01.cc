
#include <iostream>
#include <fstream>
#include <sstream>
#include <deal.II/base/conditional_ostream.h>

#include <dealiiqc/atom/parse_atom_data.h>
#include <dealiiqc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;

// A class to test ParseAtomData's central function parse
template <int dim>
void test_parse(const MPI_Comm &mpi_communicator, std::istream &is)
{
  unsigned int n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
               this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator));

  std::vector<Atom<dim>> atoms;
  std::vector<dealiiqc::types::charge> charges;
  std::vector<double> masses;

  ParseAtomData<dim> parsing_object;

  parsing_object.parse(is, atoms, charges, masses);

  for(unsigned int p = 0; p < n_mpi_processes; ++p)
    {
      MPI_Barrier(mpi_communicator);

      if (p==this_mpi_process)
        {
          // Check that the number of atoms, charges and masses parsed and added
          // are same each time we run this test.
          std::cout << "This is process: " << p << std::endl

                    << "The number of atoms parsed: " << atoms.size()
                    << std::endl

                    << "The number of different charged atom types parsed: "
                    << charges.size()
                    << std::endl

                    << "The number of different atom types parsed: "
                    << masses.size()
                    << std::endl;


          // For this particular test case all different atom types are charged.
          AssertThrow (charges.size()==masses.size(),
                       ExcInternalError());
        }

      MPI_Barrier(mpi_communicator);
    }

};

int main( int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc, argv, numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                                  << std::endl
          << "subsection Configure atoms"                         << std::endl
          << "  set Atom data file = " << SOURCE_DIR "/../data/16_NaCl_atom.data" << std::endl
          << "end"                                                << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      const std::string atom_data_file = config.get_atom_data_file();
      std::fstream fin (atom_data_file, std::fstream::in);

      test_parse<3>(MPI_COMM_WORLD, fin);
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
