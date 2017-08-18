
#include "../tests.h"

#include <deal.II-qc/adaptors/rol_vector_adaptor.h>
#include <deal.II-qc/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <Teuchos_RCP.hpp>

using namespace dealii;
using namespace dealiiqc;


// Check the rol::VectorAdaptor::set() and rol::VectorAdaptor::plus().


template <typename VectorType>
void test (const VectorType &given_vector)
{
  Teuchos::RCP<VectorType> given_vector_rcp (new VectorType(given_vector));

  // --- Testing the constructor
  rol::VectorAdaptor<VectorType> given_vector_rol (given_vector_rcp);
  AssertThrow (given_vector == *given_vector_rol.getVector(), ExcInternalError());


  Teuchos::RCP<VectorType> w_rcp =  Teuchos::rcp (new VectorType);
  rol::VectorAdaptor<VectorType> w_rol (w_rcp);

  // --- Testing VectorAdaptor::set()
  {
    w_rol.set(given_vector_rol);
    AssertThrow (given_vector == *w_rol.getVector(), ExcInternalError());
  }

  // --- Testing VectorAdaptor::plus()
  {
    VectorType u;
    u = given_vector;
    u *= 2.;
    w_rol.plus (given_vector_rol);
    AssertThrow (u == *w_rol.getVector(), ExcInternalError());
  }

  deallog << "OK" << std::endl << std::endl;
}



int main (int argc, char **argv)
{
  deallog.depth_console(10);

  dealii::Utilities::MPI::MPI_InitFinalize
  mpi_initialization (argc,
                      argv,
                      dealii::numbers::invalid_unsigned_int);

  try
    {
      {
        LinearAlgebraTrilinos::MPI::Vector trilinos_vector;
        trilinos_vector.reinit(complete_index_set(100), MPI_COMM_WORLD);

        // set the first vector
        for (unsigned int i=0; i<trilinos_vector.size(); ++i)
          trilinos_vector(i) = i;

        test (trilinos_vector);

      }
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
    };
}
