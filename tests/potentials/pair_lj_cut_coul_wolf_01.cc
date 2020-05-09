
// Short test to check correctness of PairLJCutCoulWolf classs'
// energy_and_gradient member function.
// The blessed vales are taken from pair_coul_wolf_02 and pair_lj_cut_03.

#include <deal.II-qc/configure/configure_qc.h>

#include <deal.II-qc/potentials/pair_lj_cut_coul_wolf.h>

#include "../tests.h"

using namespace dealiiqc;
using namespace dealii;

void
test(const double &                                       r,
     const double &                                       blessed_energy,
     const double &                                       blessed_gradient,
     std::shared_ptr<Potential::PairLJCutCoulWolfManager> lj_cut_coul_wolf_ptr)
{
  const std::pair<double, double> energy_gradient_0 =
    lj_cut_coul_wolf_ptr->energy_and_gradient(0, 1, r * r);

  AssertThrow(
    Testing::almost_equal(energy_gradient_0.first, blessed_energy, 200) &&
      Testing::almost_equal(energy_gradient_0.second, blessed_gradient, 200),
    ExcInternalError());

  const std::pair<double, double> energy_gradient_1 =
    lj_cut_coul_wolf_ptr->energy_and_gradient(1, 0, r * r);

  AssertThrow(Testing::almost_equal(energy_gradient_0.first,
                                    energy_gradient_1.first,
                                    200) &&
                Testing::almost_equal(energy_gradient_0.second,
                                      energy_gradient_1.second,
                                      200),
              ExcInternalError());

  std::cout << std::fixed << std::setprecision(8)
            << "Energy: " << energy_gradient_0.first
            << " Gradient scalar: " << energy_gradient_0.second << std::endl;
}

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss << "set Dimension = 3" << std::endl
          << "subsection Configure atoms" << std::endl
          << "  set Pair potential type = LJ Coulomb Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 0.95, 0.95" << std::endl
          << "  set Pair specific coefficients = 1, 2, 0.877, 1.55" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      auto lj_cut_coul_wolf_ptr =
        std::dynamic_pointer_cast<Potential::PairLJCutCoulWolfManager>(
          config.get_potential());

      std::shared_ptr<std::vector<dealiiqc::types::charge>> charges_ =
        std::make_shared<std::vector<dealiiqc::types::charge>>(2);
      (*charges_)[0] = 1.;
      (*charges_)[1] = -1.;

      lj_cut_coul_wolf_ptr->set_charges(charges_);

      test(0.90,
           -0.8345031730789789 + 551.3630363329171,
           1.829727499044526 - 7656.629108919712,
           lj_cut_coul_wolf_ptr);

      test(1.50, 0., 0., lj_cut_coul_wolf_ptr);

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" << std::endl;
    }

  return 0;
}
