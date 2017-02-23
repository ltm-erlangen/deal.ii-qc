/**
 * Short test to check if QC class accepts a file as input.
 * This test is a clone from energy_01.
 */

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
  Problem ( const std::istringstream &filename);
  void run ();
};

template <int dim>
Problem<dim>::Problem ( const std::istringstream &iss)
  :
  QC<dim>(iss)
{}

template <int dim>
void Problem<dim>::run()
{
  QC<dim>::run ();
  QC<dim>::write_mesh("meshout","eps");
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
      std::string filename("qc.prm"),
	          abs_path (SOURCE_DIR "/mesh_01/");
      std::istringstream iss( abs_path + " " + filename);

      Problem<3>problem( iss );
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
