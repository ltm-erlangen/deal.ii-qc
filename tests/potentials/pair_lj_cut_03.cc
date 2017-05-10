
#include <dealiiqc/configure/configure_qc.h>
#include <dealiiqc/potentials/pair_lj_cut.h>

// Short test to check validity of PairLJCut class member functions
// This test compares the partial results of pair_lj_cut_01 test to
// that of LAMMPS output.

using namespace dealiiqc;
using namespace dealii;

void test ( const double &r,
            const double &lammps_energy,
            const double &lammps_force,
            std::shared_ptr<Potential::PairLJCutManager> lj_ptr)
{
  std::pair<double, double> energy_force_0 =
    lj_ptr->energy_and_gradient( 0, 1, r*r);

  AssertThrow( fabs(energy_force_0.first-lammps_energy) < 1e5 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_0.second-lammps_force) < 1e7 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  std::pair<double, double> energy_force_1 =
    lj_ptr->energy_and_gradient( 1, 0, r*r);

  AssertThrow( fabs(energy_force_1.first-lammps_energy) < 1e5 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());
  AssertThrow( fabs(energy_force_1.second-lammps_force) < 1e7 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

  // std::cout << std::numeric_limits<double>::epsilon() << std::endl;
  // The test indicates that the computations of energy and forces are
  // differ by upto 1e-11 and 1e-9 respectively.
}

int main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc,
          argv,
          numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Pair potential type = LJ"                 << std::endl
          << "  set Pair global coefficients = 0.95 "         << std::endl
          << "  set Pair specific coefficients = 0, 1, 0.877, 1.55" << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      std::shared_ptr<Potential::PairLJCutManager> lj_ptr =
        std::static_pointer_cast<Potential::PairLJCutManager>(config.get_potential());

      // performing tests with blessed output (from LAMMPS)
      test(0.90, 551.3630363329171, 7656.629108919712, lj_ptr);
      test(1.50, 0.,                 0.,               lj_ptr);

      std::cout << "TEST PASSED!" << std::endl;
    }
  catch (...)
    {
      std::cout << "TEST FAILED!" <<std::endl;
    }

  return 0;
}
