
// Short test to check validity of PairBornCut class member functions

#include <deal.II-qc/potentials/pair_born_cut.h>

using namespace dealiiqc;
using namespace dealii;

void
test(const double &r, const double &cutoff_radius)
{
  // A, 1/RHO, SIGMA, C, and D
  std::vector<double> born_params = {0.877, 1., 1.55, 0.877, 0};

  Potential::PairBornCutManager born(cutoff_radius);

  born.declare_interactions(0,
                            1,
                            Potential::InteractionTypes::Born,
                            born_params);

  std::pair<double, double> energy_gradient_0 =
    born.energy_and_gradient(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_0.first << " "
            << "Gradient scalar value: " << energy_gradient_0.second
            << std::endl;

  std::pair<double, double> energy_gradient_1 =
    born.energy_and_gradient(1, 0, r * r);

  std::cout << "Energy: " << energy_gradient_1.first << " "
            << "Gradient scalar value: " << energy_gradient_1.second
            << std::endl;

  std::pair<double, double> energy_gradient_2 =
    born.energy_and_gradient<false>(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_2.first << " "
            << "Gradient scalar value: " << energy_gradient_2.second
            << std::endl;
}

int
main()
{
  test(0.90, 0.95);
  test(1.50, 0.95);
  test(1.55, 1.75);

  return 0;
}

/*

Maxima input script to test PairBornCutManager::energy_and_gradient()
in pair_born_cut_01.cc

Vishal Boddu 28.03.2019

Note: Second case is trivial.

energy(A, rho, S, C, D, r) := A * exp((S-r)/rho) - C / r^6 + D / r^8;

gradient(A, rho, S, C, D, r) := diff(energy(A, rho, S, C, D, r), r);

print("Energy case 1: ");
at(energy(A,  rho, S, C, D, r), [A=.877, rho=1, S=1.55, C=.877, D=0, r=.90]);
print("Gradient case 1: ");
at(gradient(A, rho, S, C, D, r), [A=.877, rho=1, S=1.55, C=.877, D=0, r=.90]);

print("Energy case 3: ");
at(energy(A, rho, S, C, D, r), [A=.877, rho=1, S=1.55, C=.877, D=0, r=1.55]);
print("Gradient case 3: ");
at(gradient(A, rho, S, C, D, r), [A=.877, rho=1, S=1.55, C=.877, D=0, r=1.55]);

*/

/*
Maxima output:

"Energy case 1: "
(%o3) "Energy case 1: "
(%o4) 0.0296990839348139
"Gradient case 1: "
(%o5) "Gradient case 1: "
(%o6) 9.321605513690633
"Energy case 3: "
(%o7) "Energy case 3: "
(%o8) 0.8137574454037673
"Gradient case 3: "
(%o9) "Gradient case 3: "
(%o10) -0.6321901112403896
*/
