
#include <dealiiqc/potentials/pair_coul_wolf.h>
#include <dealiiqc/utilities.h>

// Short test to check validity of PairLJCut class energy_and_gradient
// member function.
// The blessed file is created using maxima script: pair_coul_wolf.mc file.

using namespace dealiiqc;
using namespace dealii;

void test ( const double &r,
            const double &alpha,
            const double &cutoff_radius,
            const double &blessed_energy,
            const double &blessed_force)
{
  Potential::PairCoulWolfManager coul_wolf ( alpha, cutoff_radius);

  std::shared_ptr<std::vector<dealiiqc::types::charge> > charges_ =
    std::make_shared<std::vector<dealiiqc::types::charge>>(2);
  (*charges_)[0] =  1.;
  (*charges_)[1] = -1.;

  coul_wolf.set_charges(charges_);

  coul_wolf.declare_interactions( 0,
                                  1,
                                  Potential::InteractionTypes::Coul_Wolf);

  const std::pair<double, double> energy_force_0 =
    coul_wolf.energy_and_scalar_force( 0, 1, r*r);

  std::cout << "Energy: " << energy_force_0.first << " "
            << "Force scalar: " << energy_force_0.second << std::endl;

  AssertThrow( fabs(energy_force_0.first-blessed_energy) < 100. * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_0.second-blessed_force) < 100. * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  const std::pair<double, double> energy_force_1 =
    coul_wolf.energy_and_scalar_force( 1, 0, r*r);

  std::cout << "Energy: " << energy_force_1.first << " "
            << "Force scalar: " << energy_force_1.second << std::endl;

  AssertThrow( fabs(energy_force_1.first-blessed_energy) < 100. * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_1.second-blessed_force) < 100. * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  const std::pair<double, double> energy_force_2 =
    coul_wolf.energy_and_scalar_force<false>( 0, 1, r*r);

  std::cout << "Energy: " << energy_force_2.first << " "
            << "Force scalar: " << energy_force_2.second << std::endl;

}

int main()
{

  test(0.90, 0.25, 0.95, -0.8345031730789789, -1.829727499044526);
  test(1.50, 0.25, 0.95,  0.,                  0.               );
  test(1.00, 0.25, 1.75, -6.00939934656638,   -9.799065657452962);

  return 0;
}
