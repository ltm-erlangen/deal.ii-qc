
#include "../tests.h"

#include <deal.II-qc/adaptors/rol_vector_adaptor.h>
#include <deal.II-qc/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <boost/preprocessor/list/for_each.hpp>
#include <Teuchos_RCP.hpp>

using namespace dealii;
using namespace dealiiqc;


// Check the rol::VectorAdaptor::set() function.


// Taken from deal.II's test: parallel_vector_07
template <typename VectorType>
void prepare_vector (VectorType &v)
{
  const unsigned int
  myid    = dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD),
  numproc = dealii::Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD);

  const unsigned int set = 200;
  AssertIndexRange (numproc, set-2);
  const unsigned int local_size = set - myid;
  unsigned int global_size = 0;
  unsigned int my_start = 0;
  for (unsigned int i=0; i<numproc; ++i)
    {
      global_size += set - i;
      if (i<myid)
        my_start += set - i;
    }
  // each processor owns some indices and all
  // are ghosting elements from three
  // processors (the second). some entries
  // are right around the border between two
  // processors
  IndexSet local_owned(global_size);
  local_owned.add_range(my_start, my_start + local_size);

  // --- Prepare vector.
  v.reinit (local_owned, MPI_COMM_WORLD);
}


template <typename VectorType>
void test ()
{
  VectorType a, b, c;
  prepare_vector (a);
  prepare_vector (b);
  prepare_vector (c);

  for (auto iterator = a.begin(); iterator != a.end(); iterator++)
    *iterator = Testing::rand()/RAND_MAX*100.;

  for (auto iterator = b.begin(); iterator != b.end(); iterator++)
    *iterator = Testing::rand()/RAND_MAX*100.;

  for (auto iterator = c.begin(); iterator != c.end(); iterator++)
    *iterator = Testing::rand()/RAND_MAX*100.;

  a.compress(VectorOperation::insert);
  b.compress(VectorOperation::insert);
  c.compress(VectorOperation::insert);

  Teuchos::RCP<VectorType> a_rcp (new VectorType(a));
  Teuchos::RCP<VectorType> b_rcp (new VectorType(b));
  Teuchos::RCP<VectorType> c_rcp (new VectorType(c));

  // --- Testing the constructor
  rol::VectorAdaptor<VectorType> a_rol (a_rcp);
  rol::VectorAdaptor<VectorType> b_rol (b_rcp);
  rol::VectorAdaptor<VectorType> c_rol (c_rcp);

  Teuchos::RCP<std::ostream> out_stream;
  Teuchos::oblackholestream bhs; // outputs nothing

  if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
    out_stream = Teuchos::rcp(&std::cout, false);
  else
    out_stream = Teuchos::rcp(&bhs, false);

  a_rol.checkVector (b_rol, c_rol, true, *out_stream);
}



int main (int argc, char **argv)
{


  dealii::Utilities::MPI::MPI_InitFinalize
  mpi_initialization (argc,
                      argv,
                      1);

  unsigned int myid = dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
  deallog.push(dealii::Utilities::int_to_string(myid));


  if (myid == 0)
    {
      deallog.depth_console(10); // initlog();
      deallog << std::setprecision(4);
    }

  try
    {
      test<LinearAlgebraTrilinos::MPI::Vector>();
      test<LinearAlgebra::distributed::Vector<double>>();
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
