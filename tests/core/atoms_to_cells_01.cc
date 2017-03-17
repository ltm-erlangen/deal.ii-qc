// print out atom-to-cell association

#include <iostream>
#include <fstream>
#include <sstream>

#include <dealiiqc/qc.h>

using namespace dealii;
using namespace dealiiqc;

template <int dim>
class Problem : public QC<dim>
{
public:
  Problem (const ConfigureQC &);
  void run ();
};

template <int dim>
Problem<dim>::Problem (const ConfigureQC &config)
  :
  QC<dim>(config)
{}

template <int dim>
void Problem<dim>::run()
{
  QC<dim>::run ();

  for (auto a = QC<dim>::atoms.begin(); a != QC<dim>::atoms.end(); ++a)
    {
      QC<dim>::pcout << "x =" << a->position << "; "
                     << "ref =" << a->reference_position << "; "
                     << "cell=" << a->parent_cell->center() << std::endl;
    }
}


int main (int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                          numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 1;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description" << std::endl << std::endl
          << "5 atoms"            << std::endl << std::endl
          << "1  atom types"     << std::endl << std::endl
          << "Atoms #"        << std::endl << std::endl
          << "1 1 1 1.0 0.00 0. 0." << std::endl
          << "2 2 1 1.0 0.25 0. 0." << std::endl
          << "3 3 1 1.0 0.50 0. 0." << std::endl
          << "4 4 1 1.0 0.75 0. 0." << std::endl
          << "5 5 1 1.0 1.00 0. 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      // Define problem
      Problem<dim> problem(config);
      problem.run ();
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
