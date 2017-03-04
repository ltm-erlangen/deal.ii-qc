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
  Problem (const std::string &);
  void run ();
};

template <int dim>
Problem<dim>::Problem (const std::string &parameter_filename)
  :
  QC<dim>(parameter_filename)
{}

template <int dim>
void Problem<dim>::run()
{
  QC<dim>::run ();
  QC<dim>::pcout << QC<dim>::calculate_energy_gradient(QC<dim>::locally_relevant_displacement,
                                                       QC<dim>::gradient);
  QC<dim>::pcout << std::endl;
  QC<dim>::pcout << QC<dim>::gradient.linfty_norm() << std::endl;
}


int main (int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                          numbers::invalid_unsigned_int);

      std::string parameter_filename="qc.prm";
      std::ofstream parameter_out;
      parameter_out.open(parameter_filename.c_str(), std::ofstream::trunc);
      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 1;
      parameter_out << "set Dimension = " << dim << std::endl;
      parameter_out.close();

      // Define Problem
      Problem<dim> problem(parameter_filename);
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
