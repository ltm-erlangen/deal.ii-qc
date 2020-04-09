
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of 2 charged atoms
// interacting exclusively through Coulomb interactions.
// The blessed output is created through the script included at the end.
//
// Test case scenario is similar to that of energy_coul_wolf_01.
//
//       6------(*)       6------(*)
//      /|       |       /       /|
//     / |       |      /       / |
//    /  |       |     /       /  |
//   4   |       |    4-------5   |
//   |   2-------3    |       |   3
//   |  /       /     |       |  /
//   | /       /      |       | /
//   |/       /       |       |/
//  (*)------1       (*)------1
//
//  (*) indicates that the site has been occupied by an atom.
//  This initial hyper cube is refined once. This test is run with two processes
//  The processes own different active cells. The two charged atoms are
//  positioned such that both of them are cluster atoms. Therefore their cluster
//  weights are exactly equal to one. Both the processes have 2 energy atoms but
//  only one process should contain a non-zero sized neighbor_lists data member
//  (by the virtue of get_neighbor_lists() algorithm).



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run(const double &blessed_energy);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType>(config)
{}



template <int dim, typename PotentialType>
void
Problem<dim, PotentialType>::partial_run(const double &blessed_energy)
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();



  MPI_Barrier(QC<dim, PotentialType>::mpi_communicator);

  Testing::SequentialFileStream write_sequentially(
    QC<dim, PotentialType>::mpi_communicator);

  deallog
    << "picked up: "
    << QC<dim, PotentialType>::cell_molecule_data.cell_energy_molecules.size()
    << " number of energy atoms." << std::endl;

  const double energy = QC<dim, PotentialType>::template compute<false>(
    QC<dim, PotentialType>::locally_relevant_gradient);

  QC<dim, PotentialType>::pcout
    << "The energy computed using PairCoulWolfManager of 2 charged atom system is: "
    << energy << " eV." << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(QC<dim, PotentialType>::neighbor_lists.size(),
                                QC<dim, PotentialType>::mpi_communicator);

  QC<dim, PotentialType>::pcout << "Total number of neighbors "
                                << total_n_neighbors << std::endl;

  MPI_Barrier(QC<dim, PotentialType>::mpi_communicator);

  for (const auto &entry : QC<dim, PotentialType>::neighbor_lists)
    deallog << "   Atom I: " << entry.second.first->second.atoms[0].global_index
            << " Cluster weight: " << entry.second.first->second.cluster_weight
            << " : Atom J: "
            << entry.second.second->second.atoms[0].global_index
            << " Cluster weight: " << entry.second.second->second.cluster_weight
            << std::endl;

  AssertThrow(std::fabs(energy - blessed_energy) <
                100 * std::numeric_limits<double>::epsilon(),
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
          << "  set Number of initial global refinements = 1" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 2.255" << std::endl
          << "  set Pair potential type = Coulomb Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 2.25 " << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.26" << std::endl
          << "  set Cluster radius = 2.0" << std::endl
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
          << "1 1 1  1.0 0. 0. 0." << std::endl
          << "2 2 1  1.0 1. 1. 1." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
      problem.partial_run(1.763371185484229 /*blessed energy from Maxima*/);
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



/*
 Maxima input script for this test QC::calculate_energy_gradient()


 Algebraically defining shifted_energy;
 Verified the algebraic result with that from [Wolf et al 1999]
 Vishal Boddu 08.05.2017

 // actual code below
 <
 erfcc(r,alpha) := erfc(alpha*r)/r;

 derfcc(r,alpha) := diff( erfcc(r,alpha),r);

 derfcc_explicit(r,alpha) := -erfc(alpha*r)/r^2
                             - 2*alpha*(%e^(-alpha^2*r^2))/(sqrt(%pi)*r);

 shifted_energy(p,q,r,rc,alpha) := 14.399645*p*q*( erfcc(r,alpha)
                                   - limit( erfcc(r, alpha), r, rc)  );

 print("Energy : ");

 at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=1.,r=sqrt(3),rc=2.25,alpha=0.25]);
 >
*/
