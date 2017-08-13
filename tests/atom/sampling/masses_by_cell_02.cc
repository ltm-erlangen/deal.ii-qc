
#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



// Calculate and check the correctness of the computation of masses
// WeightsByBase::compute_inverse_masses().
//
// Derived class being used: WeightsByCell
//
// x--x--x-----x
// |  |  |     |          x  - vertices
// x--x--x o   |          o  - atom global_atom_index 6
// |  |  |     |          Only one atom among the 11 atoms is shown.
// x--x--x-----x
//




template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const std::string &s);
  void partial_run ();
};

template <int dim, typename PotentialType>
Problem<dim, PotentialType> ::Problem (const std::string &s)
  :
  QC<dim, PotentialType>(ConfigureQC(std::make_shared<std::istringstream>(s.c_str())))
{
  ConfigureQC config(std::make_shared<std::istringstream>(s.c_str()));

  QC<dim, PotentialType>::triangulation.begin_active()->set_refine_flag();
  QC<dim, PotentialType>::triangulation.execute_coarsening_and_refinement();

  QC<dim, PotentialType>::cell_molecule_data =
    CellMoleculeTools::
    build_cell_molecule_data<dim>
    (*config.get_stream(),
     QC<dim, PotentialType>::triangulation,
     config.get_ghost_cell_layer_thickness());
}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run()
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();

  LA::MPI::Vector inverse_masses;

  inverse_masses.reinit (QC<dim, PotentialType>::dof_handler.locally_owned_dofs(),
                         QC<dim, PotentialType>::locally_relevant_set,
                         QC<dim, PotentialType>::mpi_communicator,
                         true);

  QC<dim, PotentialType>::cluster_weights_method->
  compute_dof_masses (inverse_masses,
                      QC<dim, PotentialType>::dof_handler,
                      QC<dim, PotentialType>::cell_molecule_data);

  if (dealii::Utilities::MPI::n_mpi_processes(QC<dim, PotentialType>::mpi_communicator)==1)
    inverse_masses.print(std::cout);

  QC<dim, PotentialType>::pcout
      << "\n l1 norm     = " << std::setprecision(6) << inverse_masses.l1_norm ()
      << "\n l2 norm     = " << std::setprecision(6) << inverse_masses.l2_norm()
      << "\n linfty norm = " << std::setprecision(6) << inverse_masses.linfty_norm()
      << std::endl;
}


int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 2;

      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = 1."                          << std::endl
          << "    set Y center = .5"                          << std::endl
          << "    set Z center = .5"                          << std::endl
          << "    set X extent = 2."                          << std::endl
          << "    set Y extent = 1."                          << std::endl
          << "    set Z extent = 1."                          << std::endl
          << "    set X repetitions = 2"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 2.0"              << std::endl
          << "  set Pair potential type = LJ"                 << std::endl
          << "  set Pair global coefficients = 1.99 "         << std::endl
          << "  set Pair specific coefficients = 0, 0, 0.877, 1.2;" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2.01"        << std::endl
          << "  set Cluster radius = 0.2"                     << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"            << std::endl   << std::endl
          << "11 atoms"                      << std::endl   << std::endl
          << "1  atom types"                 << std::endl   << std::endl
          << "Masses"                        << std::endl   << std::endl
          << "    1   0.7"                   << std::endl   << std::endl
          << "Atoms #"                       << std::endl   << std::endl
          << "1  1  1 1.0 0.0 0.0 0."                       << std::endl
          << "2  2  1 1.0 0.5 0.0 0."                       << std::endl
          << "3  3  1 1.0 1.0 0.0 0."                       << std::endl
          << "4  4  1 1.0 2.0 0.0 0."                       << std::endl
          << "5  5  1 1.0 0.0 0.5 0."                       << std::endl
          << "6  6  1 1.0 0.5 0.5 0."                       << std::endl
          << "7  7  1 1.0 1.1 0.5 0."                       << std::endl
          << "8  8  1 1.0 0.0 1.0 0."                       << std::endl
          << "9  9  1 1.0 0.5 1.0 0."                       << std::endl
          << "10 10 1 1.0 1.0 1.0 0."                       << std::endl
          << "11 11 1 1.0 2.0 1.0 0."                       << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      const std::string s = oss.str();

      Problem<dim, Potential::PairLJCutManager> problem(s);
      problem.partial_run ();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
