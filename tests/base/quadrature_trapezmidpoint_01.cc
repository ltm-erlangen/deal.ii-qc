
// Test to check QTrapezWithMidpoint's quadrature points.

#include <iostream>

#include <deal.II-qc/base/quadrature_lib.h>


template <int dim>
void test ()
{
  dealiiqc::QTrapezWithMidpoint<dim> q;

  std::cout << "Dim::"
            << dim
            << ":\n";

  for (unsigned int i = 0; i < q.size(); ++i)
    std::cout << q.point(i)
              << "\t"
              << q.weight(i)
              << std::endl;
}


int main (int argc, char **argv)
{
  try
    {
      test<1>();
      test<2>();
      test<3>();
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
