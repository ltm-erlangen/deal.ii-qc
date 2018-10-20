
// Compute the energy of the system of 2 atoms interacting through
// LJ interactions and under a constant external potential field.
// This test closely resembles energy_and_gradient_lj_01 test.

#include <deal.II-qc/core/qc.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run(const double &blessed_energy, const double &blessed_gradient);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType>(config)
{
  const unsigned int material_id =
    QC<dim, PotentialType>::triangulation.begin_active()->material_id();
  AssertThrow(material_id == 0, ExcInternalError());
}



template <int dim, typename PotentialType>
void
Problem<dim, PotentialType>::partial_run(const double &blessed_energy,
                                         const double &blessed_gradient)
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();
  QC<dim, PotentialType>::initialize_external_potential_fields(0.);

  const auto &cell_energy_molecules =
    QC<dim, PotentialType>::cell_molecule_data.cell_energy_molecules;

  QC<dim, PotentialType>::pcout << "The number of energy atoms in the system: "
                                << cell_energy_molecules.size() << std::endl;

  QC<dim, PotentialType>::pcout << "Neighbor lists: " << std::endl;

  for (auto entry : QC<dim, PotentialType>::neighbor_lists)
    std::cout << "Atom I: " << entry.second.first->second.atoms[0].global_index
              << " "
              << "Atom J: " << entry.second.second->second.atoms[0].global_index
              << std::endl;

  const double energy = QC<dim, PotentialType>::template compute<true>(
    QC<dim, PotentialType>::locally_relevant_gradient);

  QC<dim, PotentialType>::pcout
    << "The energy computed using PairLJCutManager of 2 atom system is: "
    << energy << " eV" << std::endl;


  AssertThrow(Testing::almost_equal(energy, blessed_energy, 5),
              ExcInternalError());

  const double gradient = QC<dim, PotentialType>::locally_relevant_gradient(0);

  AssertThrow(Testing::almost_equal(gradient, blessed_gradient, 5),
              ExcInternalError());
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = .5" << std::endl
          << "    set Y center = .5" << std::endl
          << "    set Z center = .5" << std::endl
          << "    set X extent = 1." << std::endl
          << "    set Y extent = 1." << std::endl
          << "    set Z extent = 1." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 2.0" << std::endl
          << "  set Pair potential type = LJ" << std::endl
          << "  set Pair global coefficients = 1.99 " << std::endl
          << "  set Pair specific coefficients = 0, 0, 0.877, 1.55;"
          << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 2.0" << std::endl
          << "end" << std::endl

          << "subsection ext_potential_material_id_0" << std::endl
          << "  set Function expression = 5." << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "2 atoms" << std::endl
          << std::endl
          << "1  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1  1.0 0.0 0. 0." << std::endl
          << "2 2 1  1.0 1.0 0. 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairLJCutManager> problem(config);
      problem.partial_run(154.324376994195, 1877.831410474777
                          /*blessed values from Maxima*/);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}
