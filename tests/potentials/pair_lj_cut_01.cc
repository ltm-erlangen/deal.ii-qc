
#include <utility>

#include <dealiiqc/potentials/pair_lj_cut.h>
#include <dealiiqc/atom/atom.h>


// Short test to check validity of PairLJCut class member functions

using namespace dealiiqc;

int main()
{

  Atom<3> a;
  a.position = Point<3>(1.,1.,1.);
  a.type = 0;

  Atom<3> b;
  b.position = Point<3>(2.,1.,1.);
  b.type = 1;

  std::vector<double> lj_params = { 1., 1.};

  Potential::PairLJCutManager lj ( 0.9);
  lj.declare_interactions( a.type,
                           b.type,
                           Potential::InteractionTypes::LJ,
                           lj_params);
  std::pair<double, double> energy_force =
    lj.energy_and_scalar_force( a.type, b.type, (a.position-b.position).norm());

  std::cout << "Energy: " << energy_force.first << " "
            << "Force scalar: " << energy_force.second << std::endl;

  //---------------------------------------------------------------//
  Potential::PairLJCutManager lj_another ( 6.);
  lj_another.declare_interactions( a.type,
                                   b.type,
                                   Potential::InteractionTypes::LJ,
                                   lj_params);
  auto energy_force_another =
    lj_another.energy_and_scalar_force<true>( a.type, b.type, (a.position-b.position).norm());

  std::cout << "Energy: " << energy_force_another.first << " "
            << "Force scalar: " << energy_force_another.second << std::endl;

  //---------------------------------------------------------------//
  auto only_energy =
    lj_another.energy_and_scalar_force<false>( a.type, b.type, (a.position-b.position).norm());

  std::cout << "Energy: " << only_energy.first << " "
            << "Force scalar: " << only_energy.second << std::endl;

  return 0;
}
