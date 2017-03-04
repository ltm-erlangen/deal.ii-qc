

#include <iostream>
#include <fstream>
#include <sstream>

#include <dealiiqc/qc.h>
#include <deal.II/base/parameter_handler.h>

int main (int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dealiiqc;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                                  numbers::invalid_unsigned_int);
      // if no input provided
      AssertThrow(argc > 1,ExcMessage("Parameter file is required as an input argument"));

      std::string parameter_filename = argv[1];
      ParameterHandler prm;
      ConfigureQC::declare_parameters(prm);
      prm.read_input(parameter_filename,"dummy");

      const unsigned int dim = prm.get_integer("Dimension");

      if (dim == 2)
        {
          QC<2> problem(parameter_filename);
          problem.run ();
        }
      else if (dim == 3)
        {
          QC<3> problem(parameter_filename);
          problem.run ();
        }
      else if (dim ==1)
        {
          QC<1> problem(parameter_filename);
          problem.run ();
        }
      else
        Assert(false, ExcNotImplemented());

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

      return 1;
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
      return 1;
    }

  return 0;
}
