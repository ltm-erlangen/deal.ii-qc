// Compute the energy of the system of 4 charged atoms
// interacting exclusively through Coulomb interactions.
// The blessed output is created through the script included at the end.

#include <iostream>
#include <fstream>
#include <sstream>

#include <dealiiqc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const ConfigureQC &);
  void partial_run (const double &blessed_energy);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run(const double &blessed_energy)
{
  QC<dim, PotentialType>::pcout
      << "The number of energy atoms in the system: "
      << QC<dim, PotentialType>::atom_data.energy_atoms.size()
      << std::endl;

  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();

  QC<dim, PotentialType>::pcout << "Neighbor lists: " << std::endl;

  for ( auto entry : QC<dim, PotentialType>::neighbor_lists)
    std::cout << "Atom I: "  << entry.second.first->second.global_index  << " "
              << "Atom J: "  << entry.second.second->second.global_index << std::endl;

  const double energy = QC<dim, PotentialType>::template calculate_energy_gradient<false> (QC<dim, PotentialType>::gradient);
  QC<dim, PotentialType>::pcout
      << "The energy computed using PairCoulWolfManager of 4 charged atom system is: "
      << energy << " eV" << std::endl;

  AssertThrow (std::fabs(energy-blessed_energy) < 400. * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = .5"                          << std::endl
          << "    set Y center = .5"                          << std::endl
          << "    set Z center = .5"                          << std::endl
          << "    set X extent = 1."                          << std::endl
          << "    set Y extent = 1."                          << std::endl
          << "    set Z extent = 1."                          << std::endl
          << "    set X repetitions = 1"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum energy radius = 2.0"              << std::endl
          << "  set Pair potential type = Coulomb Wolf"       << std::endl
          << "  set Pair global coefficients = 0.25, 1.25 "   << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2."          << std::endl
          << "  set Cluster radius = 2.0"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"              << std::endl   << std::endl
          << "4 atoms"                         << std::endl   << std::endl
          << "2  atom types"                   << std::endl   << std::endl
          << "Atoms #"                         << std::endl   << std::endl
          << "1 1 1  1.0 0.00 0. 0."                          << std::endl
          << "2 2 2 -1.0 0.25 0. 0."                          << std::endl
          << "3 3 1  1.0 0.50 0. 0."                          << std::endl
          << "4 4 2 -1.0 0.75 0. 0."                          << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      // Define Problem
      // FIXME: PotentialType
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
      problem.partial_run (-111.1212060485294 /*blessed energy from Maxima*/);

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

 3*at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=.25,rc=1.25,alpha=0.25]) +
 2*at(shifted_energy(p,q,r,rc,alpha), [p=1.,q= 1.,r=.50,rc=1.25,alpha=0.25]) +
   at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=.75,rc=1.25,alpha=0.25]);
 >
*/
