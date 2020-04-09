
#include <deal.II/base/conditional_ostream.h>

#include <deal.II-qc/atom/parse_atom_data.h>

#include <deal.II-qc/utilities.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;

// Test ParseAtomData::parse_bonds function

template <int dim, int atomicity>
void
test_parse(const MPI_Comm &, std::istream &is)
{
  std::vector<Molecule<dim, atomicity>> molecules;
  std::vector<dealiiqc::types::charge>  charges;
  std::vector<double>                   masses;
  dealiiqc::types::bond_type            bonds[atomicity][atomicity];

  ParseAtomData<dim, atomicity> parsing_object;

  parsing_object.parse(is, molecules, charges, masses, bonds);

  for (auto i = 0; i < atomicity; ++i)
    for (auto j = 0; j < atomicity; ++j)
      if (bonds[i][j] != dealiiqc::numbers::invalid_bond_value)
        std::cout << "Atom " << i << " "
                  << "Atom " << j << " "
                  << "Bond " << +bonds[i][j] << std::endl;
  // + sign promotes the unsigned char to give the actual number representation
}

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      const std::string atom_data_file =
        SOURCE_DIR "/../data/NaCl_coreshell_1x1x1_atom.data";
      std::fstream fin(atom_data_file, std::fstream::in);

      test_parse<3, 16>(MPI_COMM_WORLD, fin);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
