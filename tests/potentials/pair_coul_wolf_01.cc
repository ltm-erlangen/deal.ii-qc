
// Short test to check validity of PairLJCut class
// energy_and_gradient member function.
// The blessed file is created using maxima script at the end of the file.

#include <deal.II-qc/potentials/pair_coul_wolf.h>

using namespace dealiiqc;
using namespace dealii;

void
test(const double &r,
     const double &alpha,
     const double &cutoff_radius,
     const double &blessed_energy,
     const double &blessed_force)
{
  Potential::PairCoulWolfManager coul_wolf(alpha, cutoff_radius);

  std::shared_ptr<std::vector<dealiiqc::types::charge>> charges_ =
    std::make_shared<std::vector<dealiiqc::types::charge>>(2);
  (*charges_)[0] = 1.;
  (*charges_)[1] = -1.;

  coul_wolf.set_charges(charges_);

  coul_wolf.declare_interactions(0, 1, Potential::InteractionTypes::Coul_Wolf);

  const std::pair<double, double> energy_gradient_0 =
    coul_wolf.energy_and_gradient(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_0.first << " "
            << "Force scalar: " << energy_gradient_0.second << std::endl;

  AssertThrow(fabs(energy_gradient_0.first - blessed_energy) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
  AssertThrow(fabs(energy_gradient_0.second - blessed_force) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());

  const std::pair<double, double> energy_gradient_1 =
    coul_wolf.energy_and_gradient(1, 0, r * r);

  std::cout << "Energy: " << energy_gradient_1.first << " "
            << "Force scalar: " << energy_gradient_1.second << std::endl;

  AssertThrow(fabs(energy_gradient_1.first - blessed_energy) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
  AssertThrow(fabs(energy_gradient_1.second - blessed_force) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());

  const std::pair<double, double> energy_gradient_2 =
    coul_wolf.energy_and_gradient<false>(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_2.first << " "
            << "Force scalar: " << energy_gradient_2.second << std::endl;
}

int
main()
{
  test(0.90, 0.25, 0.95, -0.8345031730789789, 1.829727499044526);
  test(1.50, 0.25, 0.95, 0., 0.);
  test(1.00, 0.25, 1.75, -6.00939934656638, 9.799065657452962);

  return 0;
}

// /*
// Maxima input script to test PairCoulWolf::energy_and_gradient()
// in pair_coul_wolf_01.cc

// Algebraically defining shifted_energy and shifted_gradient;
// Verified the algebraic result with that from [Wolf et al 1999]

// Vishal Boddu 28.04.2017

// */

// /* erfcc */
// erfcc(r,alpha) := erfc(alpha*r)/r;

// /* Differentiate erfcc */
// derfcc(r,alpha) := diff( erfcc(r,alpha),r);

// /* Differentiate ERFCC by hand*/
// derfcc_explicit(r,alpha) := -erfc(alpha*r)/r^2 -
// 2*alpha*(%e^(-alpha^2*r^2))/(sqrt(%pi)*r);

// /* shifted_energy and shifted_gradient */
// shifted_energy(p,q,r,rc,alpha) := 14.399645*p*q*( erfcc(r,alpha) - limit(
// erfcc(r, alpha), r, rc)  ); shifted_gradient(p,q,r,rc,alpha)
// := 14.399645*p*q*( derfcc_explicit(r,alpha) - limit(
// derfcc_explicit(r,alpha), r, rc)  );

// print("Energy case 1: ");
// at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=.9,rc=.95,alpha=0.25]);
// print("Force case 1: ");
// at(shifted_gradient(p,q,r,rc,alpha), [p=1.,q=-1.,r=.9,rc=.95,alpha=0.25]);
// float(%);
// print("Energy case 2: ");
// at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=1.,rc=1.75,alpha=0.25]);
// print("Force case 2: ");
// (at(shifted_gradient(p,q,r,rc,alpha), [p=1.,q=-1.,r=1.,rc=1.75,alpha=0.25]));
// float(%);
