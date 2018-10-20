
#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/core/qc.h>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;


template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem(const ConfigureQC &);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType>(config)
{
  GridOut().write_msh(QC<dim, PotentialType>::triangulation, std::cout);
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 2;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 5." << std::endl
          << "    set Y center = 5." << std::endl
          << "    set Z center = 5." << std::endl
          << "    set X extent = 10" << std::endl
          << "    set Y extent = 10" << std::endl
          << "    set Z extent = 10" << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 1" << std::endl
          << "end" << std::endl

          << "subsection A priori refinement" << std::endl
          << "  set Error indicator function = "
             "exp(-(x-5)*(x-5) - (y-10)*(y-10))"
          << std::endl
          << "  set Marking strategy = FixedFraction" << std::endl
          << "  set Refinement parameter = 0.25" << std::endl
          << "  set Number of refinement cycles = 3" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "3 atoms" << std::endl
          << std::endl
          << "2  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1  1.0 0.0 0.0 0." << std::endl
          << "2 2 2 -1.0 1.0 0.0 0." << std::endl
          << "3 3 1  1.0 2.0 0.0 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
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
