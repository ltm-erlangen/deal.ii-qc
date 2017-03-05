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
      oss << "set Dimension = " << dim << std::endl;
      std::istringstream prm_stream (oss.str().c_str());
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
