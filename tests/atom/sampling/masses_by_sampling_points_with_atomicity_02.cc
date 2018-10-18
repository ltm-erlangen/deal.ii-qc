
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
// x--x--x     |
// |  |  |     |
// x--x--x-----x
//
// The atomisitic system consists of 11 molecules with 3 atoms each.
// All the molecules are picked up as cluster molecules therefore all the
// molecules have cluster weight 1.
// This test computes masses for all the DoFs using cluster weights generated
// from WeightsBySamplingPoints.
// Due to the presence of hanging nodes, the supporting nodes of the hanging
// nodes get more masses than the rest.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem (const std::string &s)
    :
    QC<dim, PotentialType, atomicity>(ConfigureQC(std::make_shared<std::istringstream>(s.c_str())))
  {
    ConfigureQC config(std::make_shared<std::istringstream>(s.c_str()));

    this->triangulation.begin_active()->set_refine_flag();
    this->triangulation.execute_coarsening_and_refinement();
    this->triangulation.setup_ghost_cells();

    this->cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim, atomicity> (*config.get_stream(),
                                                this->triangulation,
                                                 GridTools::Cache<dim>(this->triangulation));
  }
  void partial_run ();
};



template <int dim, typename PotentialType, int atomicity>
void Problem<dim, PotentialType, atomicity>::partial_run()
{
  this->setup_cell_energy_molecules();
  this->setup_system();

  for (const auto &cell_molecule : this->cell_molecule_data.cell_energy_molecules)
    this->pcout << "Cell: "
                << cell_molecule.first                                 << "\t"
                << "Molecule ref position inside cell: "
                << std::fixed
                << std::setprecision(1)
                << cell_molecule.second.position_inside_reference_cell << "\t"
                << "Cluster weight: "
                << cell_molecule.second.cluster_weight
                << std::endl;

  this->pcout << std::endl;

  // setup_system() must have prepared the inverse_mass_matrix.
  auto &masses = this->inverse_mass_matrix.get_vector();

  // Get masses for comparison with blessed output.
  for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
    for (auto
         entry  = masses.block(atom_stamp).begin();
         entry != masses.block(atom_stamp).end();
         entry++)
      *entry = 1./(*entry);

  masses.compress(VectorOperation::insert);

  if (dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator)==1)
    masses.print(std::cout);

  this->pcout
      << "\n l1 norm     = " << std::setprecision(6) << masses.l1_norm ()
      << "\n l2 norm     = " << std::setprecision(6) << masses.l2_norm()
      << "\n linfty norm = " << std::setprecision(6) << masses.linfty_norm()
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

      //if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD)==0)
      deallog.depth_console (10);

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
          << "  set Pair specific coefficients = 0, 1, 0.877, 1.2;" << std::endl
          << "  set Pair specific coefficients = 0, 2, 0.877, 1.2;" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = -1"          << std::endl
          << "  set Cluster radius = .2"                      << std::endl
          << "  set Cluster weights by type = SamplingPoints" << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"            << std::endl   << std::endl
          << "33 atoms"                      << std::endl   << std::endl
          << "3  atom types"                 << std::endl   << std::endl
          << "Masses"                        << std::endl   << std::endl
          << "    1   0.7"                   << std::endl   << std::endl
          << "    2   0.23"                  << std::endl   << std::endl
          << "    3   0.43"                  << std::endl   << std::endl
          << "Atoms #"                       << std::endl   << std::endl

          << "1   1  1 1.0   0.0 0.0 0."     << std::endl
          << "2   1  2 1.0   0.1 0.0 0."     << std::endl
          << "3   1  3 1.0   0.0 0.1 0."     << std::endl

          << "4   2  1 1.0   0.5 0.0 0."     << std::endl
          << "5   2  2 1.0   0.6 0.0 0."     << std::endl
          << "6   2  3 1.0   0.5 0.1 0."     << std::endl

          << "7   3  1 1.0   1.0 0.0 0."     << std::endl
          << "8   3  2 1.0   1.1 0.0 0."     << std::endl
          << "9   3  3 1.0   1.0 0.1 0."     << std::endl

          << "10  4  1 1.0   2.0 0.0 0."     << std::endl
          << "11  4  2 1.0   1.9 0.0 0."     << std::endl
          << "12  4  3 1.0   2.0 0.1 0."     << std::endl

          << "13  5  1 1.0   0.0 0.5 0."     << std::endl
          << "14  5  2 1.0   0.1 0.5 0."     << std::endl
          << "15  5  3 1.0   0.0 0.6 0."     << std::endl

          << "16  6  1 1.0   0.5 0.5 0."     << std::endl
          << "17  6  2 1.0   0.6 0.5 0."     << std::endl
          << "18  6  3 1.0   0.5 0.6 0."     << std::endl

          << "19  7  1 1.0   1.1  0.5  0."   << std::endl
          << "20  7  2 1.0   1.11 0.5  0."   << std::endl
          << "21  7  3 1.0   1.1  0.51 0."   << std::endl

          << "22  8  1 1.0   0.0 1.0 0."     << std::endl
          << "23  8  2 1.0   0.1 1.0 0."     << std::endl
          << "24  8  3 1.0   0.0 0.9 0."     << std::endl

          << "25  9  1 1.0   0.5  1.0 0."    << std::endl
          << "26  9  2 1.0   0.51 1.0 0."    << std::endl
          << "27  9  3 1.0   0.5  0.9 0."    << std::endl

          << "28 10  1 1.0   1.0 1.0 0."     << std::endl
          << "29 10  2 1.0   1.1 1.0 0."     << std::endl
          << "30 10  3 1.0   1.0 0.9 0."     << std::endl

          << "31 11  1 1.0   2.0 1.0 0."     << std::endl
          << "32 11  2 1.0   1.9 1.0 0."     << std::endl
          << "33 11  3 1.0   2.0 0.9 0."     << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      const std::string s = oss.str();

      Problem<dim, Potential::PairLJCutManager, 3> problem (s);
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
