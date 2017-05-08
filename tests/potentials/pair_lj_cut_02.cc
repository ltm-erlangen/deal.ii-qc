
#include <dealiiqc/potentials/pair_lj_cut.h>

// Short test to check validity of PairLJCut class member functions
// This test compares the results of pair_lj_cut_01 test to
// that of LAMMPS output.

using namespace dealiiqc;
using namespace dealii;

void test ( const double &r,
            const double &cutoff_radius,
            const double &lammps_energy,
            const double &lammps_force )
{
  std::vector<double> lj_params = { 0.877, 1.55};

  Potential::PairLJCutManager lj ( cutoff_radius);
  lj.declare_interactions( 0,
                           1,
                           Potential::InteractionTypes::LJ,
                           lj_params);
  std::pair<double, double> energy_force_0 =
    lj.energy_and_gradient( 0, 1, r*r);

  AssertThrow( fabs(energy_force_0.first-lammps_energy) < 1e5 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_0.second-lammps_force) < 1e7 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  std::pair<double, double> energy_force_1 =
    lj.energy_and_gradient( 1, 0, r*r);

  AssertThrow( fabs(energy_force_1.first-lammps_energy) < 1e5 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_1.second-lammps_force) < 1e7 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  // std::cout << std::numeric_limits<double>::epsilon() << std::endl;
  // The test indicates that the computations of energy and forces are
  // differ by upto 1e-11 and 1e-9 respectively.
}

int main()
{
  try
    {
      // performing tests with blessed output (from LAMMPS)
      test(0.90, 0.95, 551.3630363329171, 7656.629108919712);
      test(1.50, 0.95,   0.,                 0.);
      test(1.55, 1.75,  -0.877,              1.20328831753208e-13);

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" <<std::endl;
    }

  return 0;
}
