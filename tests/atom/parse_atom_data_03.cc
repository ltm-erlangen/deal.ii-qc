
#include <deal.II/base/conditional_ostream.h>

#include <deal.II-qc/atom/parse_atom_data.h>

#include <deal.II-qc/utilities.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;

// Test ParseAtomData::parse_atoms function's atom type sorting

template <int dim, int atomicity>
void
test_parse(const MPI_Comm &mpi_communicator, std::istream &is)
{
  unsigned int n_mpi_processes(
    dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator));

  // In this test, molecules are mono-atomic.
  std::vector<Molecule<dim, atomicity>> molecules;
  std::vector<dealiiqc::types::charge>  charges;
  std::vector<double>                   masses;
  dealiiqc::types::bond_type            bonds[atomicity][atomicity];

  ParseAtomData<dim, atomicity> parsing_object;

  parsing_object.parse(is, molecules, charges, masses, bonds);

  for (unsigned int p = 0; p < n_mpi_processes; ++p)
    {
      MPI_Barrier(mpi_communicator);

      if (p == this_mpi_process)
        {
          // Check that the number of atoms, charges and masses parsed and added
          // are same each time we run this test.
          std::cout << "This is process: " << p << std::endl

                    << "The number of different charged atom types parsed: "
                    << charges.size() << std::endl

                    << "The number of different atom types parsed: "
                    << masses.size() << std::endl;

          for (const auto &molecule : molecules)
            for (const auto &atom : molecule.atoms)
              std::cout << "Molecule location: "
                        << molecule_initial_location(molecule)
                        << " Atom: " << atom.global_index << " Atom position "
                        << atom.position << std::endl;

          // For this particular test case all different atom types are charged.
          AssertThrow(charges.size() == masses.size(), ExcInternalError());
        }

      MPI_Barrier(mpi_communicator);
    }
}

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      const std::string atom_data_file =
        SOURCE_DIR "/../data/02_disordered_atom.data";
      std::fstream fin(atom_data_file, std::fstream::in);

      test_parse<3, 3>(MPI_COMM_WORLD, fin);
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
