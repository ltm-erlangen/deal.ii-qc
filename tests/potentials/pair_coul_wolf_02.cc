
// Short test to check validity of PairLJCut class energy_and_gradient
// member function.
// The blessed file is created using maxima script: pair_coul_wolf.mc file.

#include <deal.II-qc/configure/configure_qc.h>

#include <deal.II-qc/potentials/pair_coul_wolf.h>

using namespace dealiiqc;
using namespace dealii;

void
test(const double &                                  r,
     const double &                                  blessed_energy,
     const double &                                  blessed_gradient,
     std::shared_ptr<Potential::PairCoulWolfManager> coulf_wolf_ptr)
{
  std::shared_ptr<std::vector<dealiiqc::types::charge>> charges_ =
    std::make_shared<std::vector<dealiiqc::types::charge>>(2);
  (*charges_)[0] = 1.;
  (*charges_)[1] = -1.;

  coulf_wolf_ptr->set_charges(charges_);

  coulf_wolf_ptr->declare_interactions(0,
                                       1,
                                       Potential::InteractionTypes::Coul_Wolf);

  const std::pair<double, double> energy_gradient_0 =
    coulf_wolf_ptr->energy_and_gradient(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_0.first << " "
            << "Force scalar: " << energy_gradient_0.second << std::endl;

  AssertThrow(fabs(energy_gradient_0.first - blessed_energy) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
  AssertThrow(fabs(energy_gradient_0.second - blessed_gradient) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());

  const std::pair<double, double> energy_gradient_1 =
    coulf_wolf_ptr->energy_and_gradient(1, 0, r * r);

  std::cout << "Energy: " << energy_gradient_1.first << " "
            << "Force scalar: " << energy_gradient_1.second << std::endl;

  AssertThrow(fabs(energy_gradient_1.first - blessed_energy) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
  AssertThrow(fabs(energy_gradient_1.second - blessed_gradient) <
                100. * std::numeric_limits<double>::epsilon(),
              ExcInternalError());

  const std::pair<double, double> energy_gradient_2 =
    coulf_wolf_ptr->energy_and_gradient<false>(0, 1, r * r);

  std::cout << "Energy: " << energy_gradient_2.first << " "
            << "Force scalar: " << energy_gradient_2.second << std::endl;
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
          << "  set Pair potential type = Coulomb Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 0.95 " << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      std::shared_ptr<Potential::PairCoulWolfManager> coul_wolf_ptr =
        std::static_pointer_cast<Potential::PairCoulWolfManager>(
          config.get_potential());

      test(0.90, -0.8345031730789789, 1.829727499044526, coul_wolf_ptr);
      test(1.50, 0., 0., coul_wolf_ptr);

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" << std::endl;
    }

  return 0;
}
