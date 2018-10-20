
// Short test to check validity of PairLJCut class member functions

#include <deal.II-qc/potentials/pair_lj_cut.h>

using namespace dealiiqc;
using namespace dealii;

void
test(const double &r, const double &cutoff_radius)
{
  std::vector<double> lj_params = {0.877, 1.55};

  Potential::PairLJCutManager lj(cutoff_radius);
  lj.declare_interactions(0, 1, Potential::InteractionTypes::LJ, lj_params);
  std::pair<double, double> energy_force_0 =
    lj.energy_and_gradient(0, 1, r * r);

  std::cout << "Energy: " << energy_force_0.first << " "
            << "Force scalar: " << energy_force_0.second << std::endl;

  std::pair<double, double> energy_force_1 =
    lj.energy_and_gradient(1, 0, r * r);

  std::cout << "Energy: " << energy_force_1.first << " "
            << "Force scalar: " << energy_force_1.second << std::endl;

  std::pair<double, double> energy_force_2 =
    lj.energy_and_gradient<false>(0, 1, r * r);

  std::cout << "Energy: " << energy_force_2.first << " "
            << "Force scalar: " << energy_force_2.second << std::endl;
}

int
main()
{
  test(0.90, 0.95);
  test(1.50, 0.95);
  test(1.55, 1.75);

  return 0;
}
