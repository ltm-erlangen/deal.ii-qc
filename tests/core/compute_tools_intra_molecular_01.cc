
#include "../tests.h"

#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/potentials/pair_lj_cut.h>

using namespace dealiiqc;
using namespace dealii;



// Short test to check the correctness of ComputeTools::energy_and_gradient()
// that calculates the intra molecular energy and gradient within a molecule
// whose atoms interact through LJ potential.
// The blessed output is taken from pair_lj_cut_01 and
// compute_tools_inter_atomic_01 test.



template <int spacedim>
void test (const double &r,
           const double &cutoff_radius)
{
  std::vector<double> lj_params = { 0.877, 1.55};

  Potential::PairLJCutManager lj ( cutoff_radius);
  lj.declare_interactions( 0,
                           1,
                           Potential::InteractionTypes::LJ,
                           lj_params);

  Atom<spacedim> atom_1, atom_2;

  atom_1.type = 0;
  atom_1.position =
    (spacedim==3) ? Point<spacedim>(1.12+r, 2.4, 7.3) :
    (spacedim==2  ? Point<spacedim>(1.12+r, 2.4     ) : Point<spacedim>(1.12+r));

  atom_2.type = 1;
  atom_2.position =
    (spacedim==3) ? Point<spacedim>(1.12, 2.4, 7.3) :
    (spacedim==2  ? Point<spacedim>(1.12, 2.4     ) : Point<spacedim>(1.12));

  Tensor<1, spacedim> rij = (atom_1.position - atom_2.position);

  std::pair<double, double> lj_energy_gradient_0 =
    lj.energy_and_gradient( 0, 1, r*r);

  const Tensor<1, spacedim> lj_gradient_0 = rij * lj_energy_gradient_0.second
                                            /
                                            rij.norm();

  std::cout << "Energy: "       << lj_energy_gradient_0.first  << " "
            << "Force scalar: " << lj_energy_gradient_0.second << std::endl;

  std::pair<double, double> lj_energy_gradient_1 =
    lj.energy_and_gradient( 1, 0, r*r);

  const Tensor<1, spacedim> lj_gradient_1 = -rij * lj_energy_gradient_1.second
                                            /
                                            rij.norm();

  std::cout << "Energy: "       << lj_energy_gradient_1.first  << " "
            << "Force scalar: " << lj_energy_gradient_1.second << std::endl;

  // --- Actual testing starts from here.

  Molecule<spacedim, 2> molecule;
  molecule.atoms[0] = atom_1;
  molecule.atoms[1] = atom_2;

  std::pair<double, std::array<Tensor<1, spacedim>, 2> >
  intra_molecular = ComputeTools::energy_and_gradient (lj,
                                                       molecule);

  AssertThrow (Testing::almost_equal (lj_energy_gradient_0.first,
                                      intra_molecular.first,
                                      50),
               ExcInternalError());

  for (int d = 0; d < spacedim; d++)
    {
      AssertThrow (Testing::almost_equal (lj_gradient_0[d],
                                          intra_molecular.second[0][d],
                                          50),
                   ExcInternalError());

      AssertThrow (Testing::almost_equal (lj_gradient_1[d],
                                          intra_molecular.second[1][d],
                                          50),
                   ExcInternalError());

      AssertThrow (Testing::almost_equal (lj_gradient_0[d],
                                          -lj_gradient_1[d],
                                          50),
                   ExcInternalError());
    }
}

int main()
{

  test<1>(0.90, 0.95);
  test<1>(1.50, 0.95);

  test<2>(0.90, 0.95);
  test<2>(1.50, 0.95);

  test<3>(0.90, 0.95);
  test<3>(1.50, 0.95);

  return 0;
}
