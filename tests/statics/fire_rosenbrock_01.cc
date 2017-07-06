
#include <deal.II-qc/statics/fire.h>

#include <deal.II/base/utilities.h>


using namespace dealii;
using namespace dealiiqc;



// Test to verify correctness of SolverFIRE::sovle()
// The objective function is the extended Rosenbrock function.
// The Rosenbrock function is a non-convex function used as a test problem
// for optimization algorithms introduced by Howard H. Rosenbrock.
//
// f(X) = f(x_0, x_1, ..., x_{N-1})
//
//      = \sum_{i=0}^{\frac{N}{2} -1}
//
//        \left[
//                  a ( x_{2i}^2 - x_{2i+1} )^2
//                  +
//                  b ( x_{2i}   - 1        )^2
//        \right],
//
//   where N is even and a = 100 and b = 1.
//
// DOI: 10.1007/BF02196600




using vector_t = typename dealii::Vector<double>;


double compute (vector_t &G, const vector_t &X)
{
  AssertThrow (X.size() % 2 == 0,
               ExcInternalError());

  double value = 0.;

  // Value of the objective function.
  for (unsigned int i = 0; i < X.size()/2; ++i)
    value += 100 *
             dealii::Utilities::fixed_power<2>( X(2*i) * X(2*i) - X(2*i+1) )
             +
             dealii::Utilities::fixed_power<2>( X(2*i)          -       1  );

  // Gradient of the objective function.
  for (unsigned int i = 0; i < X.size()/2; ++i)
    {
      G(2*i)   = ( X(2*i) * X(2*i) - X(2*i+1) ) * X(2*i) * 400
                 +
                 ( X(2*i)          -       1  ) *            2;

      G(2*i+1) = ( X(2*i) * X(2*i) - X(2*i+1) ) *         -200;
    }

  return value;
}



void test (const unsigned int N,
           const double tol)
{
  AssertThrow (N % 2 == 0,
               ExcInternalError());

  vector_t X (N);

  // Use this to initialize DiagonalMatrix
  X = 1.;

  // Create inverse diagonal matrix.
  DiagonalMatrix<vector_t> inv_mass;
  inv_mass.reinit(X);

  // Set initial guess.
  for (unsigned int i=0; i < N/2; i++)
    {
      X(2*i)   = -1.2;
      X(2*i+1) =  1.0;
    }

  auto additional_data =
    statics::SolverFIRE<vector_t>::AdditionalData(1, 1, 1);

  SolverControl solver_control (1e05, tol);

  statics::SolverFIRE<vector_t> fire (solver_control, additional_data);

  fire.solve(compute, X, inv_mass);

  X.print(std::cout);
}



int main ()
{
  test (  2, 1e-15 );
  test ( 10, 1e-15 );
  test ( 20, 1e-15 );

}
