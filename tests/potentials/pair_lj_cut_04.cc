
// Short test to check validity of PairLJCut class member functions
// This test compares the results of pair_lj_cut_01 test to
// that of LAMMPS output.

#include "../tests.h"

#include <deal.II-qc/potentials/pair_lj_cut.h>

using namespace dealiiqc;
using namespace dealii;

void test (const double r,
           const double cutoff_radius,
           const bool   with_tail = false)
{
  std::vector<double> lj_params = {0.877, .5};

  Potential::PairLJCutManager lj (cutoff_radius, with_tail);
  lj.declare_interactions( 0,
                           1,
                           Potential::InteractionTypes::LJ,
                           lj_params);

  std::pair<double, double> energy_gradient_0 =
    lj.energy_and_gradient( 0, 1, r*r);

  std::pair<double, double> energy_gradient_1 =
    lj.energy_and_gradient( 1, 0, r*r);

  AssertThrow (Testing::almost_equal (energy_gradient_0.first,
                                      energy_gradient_1.first,
                                      20)
               &&
               Testing::almost_equal (energy_gradient_0.second,
                                      energy_gradient_1.second,
                                      20),
               ExcInternalError())

  std::cout << std::fixed
            << std::setprecision(8)
            << "Energy: "
            << energy_gradient_0.first
            << " Gradient scalar: "
            << energy_gradient_0.second
            << "\t";
}

int main()
{
  try
    {
      // The following tests evaluate values around cutoff radius.
      for (int i = -5; i < 5; ++i)
        {
          test(2.95 + 0.0001*i, 2.95, false);
          test(2.95 + 0.0001*i, 2.95, true);
          std::cout << std::endl;
        }
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" <<std::endl;
    }

  return 0;
}
