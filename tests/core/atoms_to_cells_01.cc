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
  Problem ();
  void run ();
};

template <int dim>
Problem<dim>::Problem ()
  :
  QC<dim>()
{}

template <int dim>
void Problem<dim>::run()
{
  {
    GridGenerator::hyper_cube (QC<dim>::triangulation);
    QC<dim>::triangulation.refine_global(1);
  }

  QC<dim>::setup_system();
  QC<dim>::associate_atoms_with_cells();


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

      Problem<1> problem;
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
