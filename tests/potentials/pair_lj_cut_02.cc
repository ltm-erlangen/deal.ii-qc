
#include <dealiiqc/potentials/pair_lj_cut.h>

// Short test to check validity of PairLJCut class member functions

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
    lj.energy_and_scalar_force( 0, 1, r*r);

  AssertThrow( fabs(energy_force_0.first-lammps_energy) < 100.*std::numeric_limits<double>::min(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_0.second-lammps_force) < 100.*std::numeric_limits<double>::min(),
               ExcInternalError());
  std::pair<double, double> energy_force_1 =
    lj.energy_and_scalar_force( 1, 0, r*r);

  AssertThrow( fabs(energy_force_1.first-lammps_energy) < 100.*std::numeric_limits<double>::min(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_1.second-lammps_force) < 100.*std::numeric_limits<double>::min(),
               ExcInternalError());

}

int main()
{
  try
    {
      //TODO
      //test(0.90, 0.95, );
      //test(1.50, 0.95, );
      //test(1.55, 1.75, );

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" <<std::endl;
    }

  return 0;
}
