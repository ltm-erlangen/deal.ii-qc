
// same as partial_differentive_of_vector_wrt_dofs_01
// but calcualte forces using QC class for 2 Coulomb particles.
// The output is made exactly the same (forces) as in _01 test by multiplying
// the actual forces with inverse of the potential derivative.

#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/core/qc.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>

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
  QC<dim, PotentialType>::pcout << "energy      = "
                                << energy
                                << std::endl;

  // serial vector with all forces:
  const unsigned int n_dofs = QC<dim, PotentialType>::dof_handler.n_dofs();
  dealii::Vector<double> gradient(n_dofs);
  // manually copy parallel vector:
  const IndexSet locally_owned_dofs = QC<dim, PotentialType>::dof_handler.locally_owned_dofs();
  gradient = 0.;
  for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
    {
      const unsigned int ind = locally_owned_dofs.nth_index_in_set(i);
      gradient(ind) = QC<dim, PotentialType>::gradient(ind);
    }
  dealii::Utilities::MPI::sum(gradient, QC<dim, PotentialType>::mpi_communicator, gradient);

  // derivative of energy for this potential and the given distance
  // (cluster weights are 1)
  const double derivative = -6.148223356137124;
  gradient *= 1./derivative;
  QC<dim, PotentialType>::pcout << "l1 norm     = "
                                << gradient.l1_norm ()
                                << std::endl
                                << "l2 norm     = "
                                << gradient.l2_norm()
                                << std::endl
                                << "linfty norm = "
                                << gradient.linfty_norm ()
                                << std::endl;

  if (dealii::Utilities::MPI::n_mpi_processes(QC<dim, PotentialType>::mpi_communicator)==1)
    {
      for (unsigned int i = 0; i < n_dofs; i+=dim)
        {
          for (int d = 0; d < dim; ++d)
            QC<dim, PotentialType>::pcout << gradient[i+d] <<  "\t";
          QC<dim, PotentialType>::pcout << std::endl;
        }
    }
  else
    {
      // for more than one MPI cores dof numbering could be different,
      // just check l2, l1 and l infinity norm for correctness.
    }

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
