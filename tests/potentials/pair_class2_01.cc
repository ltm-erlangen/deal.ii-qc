
// Short test to check validity of PairClass2Manager class member functions

#include <deal.II-qc/potentials/pair_class2.h>

using namespace dealiiqc;
using namespace dealii;

void
test(const double &r)
{
  // r_0, k2, k3, and k4
  std::vector<double> class2_params = {0, 1.1, 2.2, 3.7};

  Potential::PairClass2Manager class2;

  class2.declare_interactions(0,
                              1,
                              Potential::InteractionTypes::Class2,
                              class2_params);

  const std::pair<double, double> energy_gradient_0 =
    class2.energy_and_gradient(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_0.first << " "
            << "Gradient scalar value: " << energy_gradient_0.second
            << std::endl;

  const std::pair<double, double> energy_gradient_1 =
    class2.energy_and_gradient(1, 0, r * r);

  std::cout << "Energy: " << energy_gradient_1.first << " "
            << "Gradient scalar value: " << energy_gradient_1.second
            << std::endl;

  const std::pair<double, double> energy_gradient_2 =
    class2.energy_and_gradient<false>(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_2.first << " "
            << "Gradient scalar value: " << energy_gradient_2.second
            << std::endl;
}

int
main()
{
  test(0.90);
  test(1.50);
  test(1.55);

  return 0;
}

/*

Maxima input script to test PairClass2Manager::energy_and_gradient()
in pair_class2_01.cc

Vishal Boddu 28.03.2019

energy(r, rm, k2, k3, k4) := k2*(r-rm)^2 + k3*(r-rm)^3 + k4*(r-rm)^4;

gradient(r, rm, k2, k3, k4) := diff(energy(r, rm, k2, k3, k4), r);

print("Energy case 1: ");
at(energy(r, rm, k2, k3, k4), [r=0.9, rm=0., k2=1.1, k3=2.2, k4=3.7]);
print("Gradient case 1: ");
at(gradient(r, rm, k2, k3, k4), [r=0.9, rm=0., k2=1.1, k3=2.2, k4=3.7]);

print("Energy case 2: ");
at(energy(r, rm, k2, k3, k4), [r=1.5, rm=0., k2=1.1, k3=2.2, k4=3.7]);
print("Gradient case 2: ");
at(gradient(r, rm, k2, k3, k4), [r=1.5, rm=0., k2=1.1, k3=2.2, k4=3.7]);

print("Energy case 3: ");
at(energy(r, rm, k2, k3, k4), [r=1.5, rm=0., k2=1.1, k3=2.2, k4=3.7]);
print("Gradient case 3: ");
at(gradient(r, rm, k2, k3, k4), [r=1.5, rm=0., k2=1.1, k3=2.2, k4=3.7]);

*/

/*
Maxima output:

"Energy case 1: "" "
(%o3)	"Energy case 1: "
(%o4)	4.922370000000001
"Gradient case 1: "" "
(%o5)	"Gradient case 1: "
(%o6)	18.1152
"Energy case 2: "" "
(%o7)	"Energy case 2: "
(%o8)	28.63125
"Gradient case 2: "" "
(%o9)	"Gradient case 2: "
(%o10)	68.10000000000001
"Energy case 3: "" "
(%o11)	"Energy case 3: "
(%o12)	28.63125
"Gradient case 3: "" "
(%o13)	"Gradient case 3: "
(%o14)	68.10000000000001
*/
