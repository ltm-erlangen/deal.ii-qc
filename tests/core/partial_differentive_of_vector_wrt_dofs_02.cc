
// same as partial_differentive_of_vector_wrt_dofs_01
// but calcualte forces using QC class for 2 Coulomb particles.
// The output is made exactly the same (forces) as in _01 test by multiplying
// the actual forces with inverse of the potential derivative.

#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const ConfigureQC &);
  void partial_run ();
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run()
{
  QC<dim, PotentialType>::setup_energy_atoms_with_cluster_weights();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();

  const double energy = QC<dim, PotentialType>::template calculate_energy_gradient<true> (QC<dim, PotentialType>::gradient);

  // serial vector with all forces:
  const unsigned int n_dofs = QC<dim, PotentialType>::dof_handler.n_dofs();
  dealii::Vector<double> gradient(n_dofs);
  gradient = QC<dim, PotentialType>::gradient;
  // derivative of energy for this potential and the given distance
  // (cluster weights are 1)
  const double derivative = -6.148223356137124;
  gradient *= 1./derivative;
  for (unsigned int i = 0; i < n_dofs; i+=dim)
    {
      for (int d = 0; d < dim; ++d)
        QC<dim, PotentialType>::pcout << gradient[i+d] <<  "\t";
      QC<dim, PotentialType>::pcout << std::endl;
    }
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
          << "  set Maximum cutoff radius = 9.0"              << std::endl
          << "  set Pair potential type = Coulomb Wolf"       << std::endl
          << "  set Pair global coefficients = 0.25, 8.25 "   << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 10.0"        << std::endl
          << "  set Cluster radius = 99.0"                    << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"              << std::endl   << std::endl
          << "2 atoms"                         << std::endl   << std::endl
          << "2  atom types"                   << std::endl   << std::endl
          << "Atoms #"                         << std::endl   << std::endl
          << "1 1 1  1.0 0.23 0.37 0."                        << std::endl
          << "2 2 2 -1.0 1.73 0.43 0."                        << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      // Define Problem
      // FIXME: PotentialType
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
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
