
#include <deal.II-qc/statics/fire.h>


using namespace dealii;
using namespace dealiiqc;



// Test to verify correctness of SolverFIRE::sovle()
// The objective function is f(x,y) = x^2 + y^2.



using vector_t = typename dealii::Vector<double>;


double compute (vector_t &G, const vector_t &X)
{
  AssertThrow (X.size() == 2 && G.size() == 2,
               ExcInternalError());

  G(0) = 2*X(0);
  G(1) = 2*X(1);

  return X.norm_sqr();
}



void test (const double x,
           const double y,
           const double tol)
{
  vector_t X;

  X.reinit(2, true);

  // Use this to initialize DiagonalMatrix
  X = 1.;

  // Create inverse diagonal matrix.
  DiagonalMatrix<vector_t> inv_mass;
  inv_mass.reinit(X);

  // Set initial iterate.
  X(0) = x;
  X(1) = y;

  auto additional_data =
    statics::SolverFIRE<vector_t>::AdditionalData(1e-3, 1e-1, 0.1);

  SolverControl solver_control (1e05, tol);

  statics::SolverFIRE<vector_t> fire (solver_control, additional_data);

  fire.solve(compute, X, inv_mass);

  X.print(std::cout);

}

int main ()
{
  test (  10,  -2, 1e-15 );
  test (-0.1, 0.1, 1e-15 );
  test ( 9.1,-6.1, 1e-15 );

}
