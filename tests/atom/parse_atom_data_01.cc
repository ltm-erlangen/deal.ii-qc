
#include "../tests.h"

#include <deal.II-qc/atom/parse_atom_data.h>

using namespace dealii;
using namespace dealiiqc;

// A class to test ParseAtomData's central function parse
template <int dim>
void test_parse(const MPI_Comm &mpi_communicator, std::istream &is)
{
  // In this test, molecules are mono-atomic.
  std::vector<Molecule<dim,1>> atoms;
  std::vector<dealiiqc::types::charge> charges;
  std::vector<double> masses;

  ParseAtomData<dim> parsing_object;

  parsing_object.parse(is, atoms, charges, masses);

  Testing::SequentialFileStream write_sequentially(mpi_communicator);

  // Check that the number of atoms, charges and masses parsed and added
  // are same each time we run this test.

  deallog << "Atoms parsed: "
          << atoms.size()
          << ". Charged atom types: "
          << charges.size()
          << ". Total atom types: "
          << masses.size()
          << std::endl;

}

int main( int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
          dealii::numbers::invalid_unsigned_int);

      const std::string atom_data_file = SOURCE_DIR "/../data/16_NaCl_atom.data";
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
