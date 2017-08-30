
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

// Molecule has three atoms



template <int spacedim, int atomicity>
void test (const double eps,
           const double rm,
           const double cutoff_radius)
{
  const std::vector<double> lj_params = {eps, rm};

  Potential::PairLJCutManager lj (cutoff_radius);
  lj.declare_interactions (0,
                           0,
                           Potential::InteractionTypes::LJ,
                           lj_params);

  Molecule<spacedim, atomicity> molecule;

  for (auto &atom : molecule.atoms)
    {
      for (int sd = 0; sd < spacedim; ++sd)
        atom.position[sd] = static_cast<double>(Testing::rand())
                            /
                            RAND_MAX;
      atom.type = 0;
    }

  // Compute intra-molecular energy and gradients.
  std::pair<double, std::array<Tensor<1, spacedim>, atomicity> >
  intra_molecular = ComputeTools::energy_and_gradient (lj,
                                                       molecule);

  std::cout << "Dim: " << spacedim << " Atomicity: " << atomicity << std::endl;

  // Log energy and gradient values.
  std::cout << "Energy: "
            << intra_molecular.first
            << std::endl
            << "Gradient: "
            << std::endl;

  for (const auto& gradient_tensor : intra_molecular.second)
    std::cout << "\t\t"
              << gradient_tensor
              << std::endl;


  // Sum over all the intra molecular forces/gradients should be zero.

  Tensor<1, spacedim> external_forces;
  external_forces = 0.;

  for (const auto& gradient_tensor : intra_molecular.second)
    external_forces += gradient_tensor;

  std::cout << "External force on molecule: "
            << -external_forces
            << std::endl;

  for (int d = 0; d < spacedim; d++)
    AssertThrow (std::fabs(external_forces[d]) < 1e-12,
                 ExcInternalError());
}

int main()
{

  test<1, 2>(10.87, 0.650, 1.95);
  test<1, 3>(21.02, 0.001, 1.88);
  test<1, 4>(31.32, 0.343, 0.88);

  test<2, 2>(10.87, 0.650, 1.95);
  test<2, 3>(21.02, 0.001, 1.88);
  test<3, 4>(31.32, 0.343, 0.88);

  test<3, 2>(10.87, 0.650, 1.95);
  test<3, 3>(21.02, 0.001, 1.88);
  test<3, 4>(31.32, 0.343, 0.88);

  return 0;
}
