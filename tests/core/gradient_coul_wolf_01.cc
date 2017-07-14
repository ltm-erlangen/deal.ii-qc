
// Check the gradient of the total energy.
// Calculate the gradient using QC class for 2 Coulomb particles.
//
// *-------o
// |       |          o,*  - vertices
// |       |          *    - atoms
// |       |          o    - dof sites at which gradient value is zero
// o-------*
//
// 4 entries of the gradient of the total energy are zeros.



#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/core/qc.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>

using namespace dealii;
using namespace dealiiqc;

// #define WRITE_GRID



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
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();

  const double energy =
    QC<dim, PotentialType>::template
    compute<true> (QC<dim, PotentialType>::locally_relevant_gradient);

  QC<dim, PotentialType>::pcout << "energy       = "
                                << energy
                                << std::endl;

  // Get total number of dofs.
  const unsigned int n_dofs = QC<dim, PotentialType>::dof_handler.n_dofs();

  // Get locally owned dofs and count the number of zero entries in gradient.
  const IndexSet locally_owned_dofs = QC<dim, PotentialType>::dof_handler.locally_owned_dofs();

  // Sum number of zero entries in gradient by going through mutually exclusive
  // locally owned dofs.
  unsigned int n_zeros = 0;
  for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
    if (QC<dim, PotentialType>::
        locally_relevant_gradient(locally_owned_dofs.nth_index_in_set(i)) == 0.)
      n_zeros++;

  // Get global number of zero entries in gradient.
  n_zeros =
    dealii::Utilities::MPI::
    sum(n_zeros, QC<dim, PotentialType>::mpi_communicator);

  // derivative of energy for this potential and the given distance
  // (cluster weights are 1)
  QC<dim, PotentialType>::pcout
      << "l1 norm      = "
      << QC<dim, PotentialType>::locally_relevant_gradient.l1_norm ()
      << std::endl
      << "l2 norm      = "
      << QC<dim, PotentialType>::locally_relevant_gradient.l2_norm()
      << std::endl
      << "linfty norm  = "
      << QC<dim, PotentialType>::locally_relevant_gradient.linfty_norm ()
      << std::endl;

  QC<dim, PotentialType>::pcout << "n_dofs       = "
                                <<  n_dofs
                                << std::endl;

  QC<dim, PotentialType>::pcout << "n_grad_zeros = "
                                <<  n_zeros
                                << std::endl;

  // For tests with more than one MPI processes dof numbering could be
  // different, so we only check l2, l1 and linfty norm for correctness.
  // For the test with one MPI process, output the gradient entries and
  // compare with blessed output.
  if (dealii::Utilities::MPI::n_mpi_processes(QC<dim, PotentialType>::mpi_communicator)==1)
    for (unsigned int i = 0; i < n_dofs; i+=dim)
      {
        for (int d = 0; d < dim; ++d)
          QC<dim, PotentialType>::pcout << QC<dim, PotentialType>::locally_relevant_gradient[i+d]
                                        <<  "\t";
        QC<dim, PotentialType>::pcout << std::endl;
      }

#ifdef WRITE_GRID
  if (dealii::Utilities::MPI::this_mpi_process(QC<dim, PotentialType>::mpi_communicator)==0)
    {
      std::map<dealii::types::global_dof_index, Point<dim> > support_points;
      DoFTools::map_dofs_to_support_points (QC<dim, PotentialType>::mapping,
                                            QC<dim, PotentialType>::dof_handler,
                                            support_points);

      const std::string filename =
        "grid" + dealii::Utilities::int_to_string(dim) + ".gp";
      std::ofstream f(filename.c_str());

      f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
        << "set output \"grid" << dealii::Utilities::int_to_string(dim) << ".png\"" << std::endl
        << "set size square" << std::endl
        << "set view equal xy" << std::endl
        << "unset xtics" << std::endl
        << "unset ytics" << std::endl
        << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
      GridOut().write_gnuplot (QC<dim, PotentialType>::triangulation, f);
      f << "e" << std::endl;

      DoFTools::write_gnuplot_dof_support_point_info(f,
                                                     support_points);
      f << "e" << std::endl;
    }
#endif

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
          << "1 1 1  1.0 0.0 1.0 0."                        << std::endl
          << "2 2 2 -1.0 1.0 0.0 0."                        << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      Problem<dim, Potential::PairCoulWolfManager> problem(prm_stream);
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


 Algebraically defining shifted_energy and its derivative;
 Verified the algebraic result with that from [Wolf et al 1999]

 // actual code below
 <
 erfcc(r,alpha) := erfc(alpha*r)/r;

 derfcc(r,alpha) := diff( erfcc(r,alpha),r);

 derfcc_explicit(r,alpha) := -erfc(alpha*r)/r^2
                             - 2*alpha*(%e^(-alpha^2*r^2))/(sqrt(%pi)*r);

 shifted_energy(p,q,r,rc,alpha) := 14.399645*p*q*( erfcc(r,alpha)
                                   - limit( erfcc(r, alpha), r, rc)  );

 grad_r(p,q,r,rc,alpha) := 14.399645*p*q*( derfcc_explicit(r,alpha)
                                          - derfcc_explicit(rc,alpha) );

 grad_x(p,q,r,rc,alpha) := grad_r(p,q,r,rc,alpha)/r;

 print("Energy : ");
 float(at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=sqrt(2),rc=8.25,alpha=0.25]));

 print("Derivative with respect to r : ");
 float(at(grad_r(p,q,r,rc,alpha), [p=1.,q=-1.,r=sqrt(2),rc=8.25,alpha=0.25]));

 print("Derivative with respect to r divided by r: ");
 float(at(grad_x(p,q,r,rc,alpha), [p=1.,q=-1.,r=sqrt(2),rc=8.25,alpha=0.25]));

 >
 // end of code
 Output:

 "Energy : "
 (%o7) "Energy : "
 (%o8) −6.276939683528505
 "Derivative with respect to r : "
 (%o9) "Derivative with respect to r : "
 (%o10) 6.969894820319487
 "Derivative with respect to r divided by r: "
 (%o11) "Derivative with respect to r divided by r: "
 (%o12) 4.928459891604901
 */
