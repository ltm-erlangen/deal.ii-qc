
// Calculate and check the correctness of the gradient using QC class for
// 2 particles embedded in a mesh consisting of a single hanging node.
// Run with one MPI process will check the correctness of the gradient,
// however due to (machine dependent) dof numbering for run with more
// than one MPI process the gradient entries are shuffled.
// Therefore, for the runs with more than one MPI process only
// l1, l2 and linfty norms are checked for correctness.
//
// o--o--x-----x
// |  |  |     |          x,o  - vertices
// x--x--o   * |          *    - atoms
// |* |  |     |          o    - dof sites at which gradient value is zero
// x--x--x-----x
//
// 6 entries of the gradient of the total energy are zeros.



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
  Problem (const std::string &);
  void partial_run ();
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const std::string &s)
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

  const auto &gradient = QC<dim, PotentialType>::locally_relevant_gradient;

  // Count the number of zero entries within  locally owned entries.
  for (auto entry = gradient.begin(); entry != gradient.end(); ++entry)
    if (*entry ==0.)
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
          QC<dim, PotentialType>::pcout
              << QC<dim, PotentialType>::locally_relevant_gradient[i+d]
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
          << "  set Cluster radius = 2.0"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"              << std::endl   << std::endl
          << "2 atoms"                         << std::endl   << std::endl
          << "1  atom types"                   << std::endl   << std::endl
          << "Atoms #"                         << std::endl   << std::endl
          << "1 1 1 1.0 0.115 0.185 0."                       << std::endl
          << "2 2 1 1.0 1.730 0.430 0."                       << std::endl;

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

/*

Maxima script:

a : [0.115, 0.185];
b : [1.73, 0.43];

r : a-b;
rnorm : float(sqrt(r.r));

eps : 0.877;
sigma : 1.2;

energy(x) := 0.877 * (sigma/x )^6 * ((sigma/x )^6 -2. );
deriv(x) := 12 * 0.877 * (sigma/x )^6 * (1 - (sigma/x)^6) / x;

factor : float(at(deriv(x), [x=rnorm]));
energy : float(at(energy(x), [x=rnorm]));

load ("eigen");
n(x) := unitvector(x);
nab : n(a-b);

N0(x,y) :=       if x > 1.0 then (2-x)*(1-y)
            else if x < 0.5 then 0
            else if y > 0.5 then 0
            else                 4*(x-.5)*(.5-y);

N1(x,y) :=       if x > 1.0 then (x-1)*(1-y)
            else                 0;

N2(x,y) :=       if x > 1.0 then (2-x)*(  y)
            else if x < 0.5 then 0
            else if y < 0.5 then 0
            else                 4*(x-.5)*(y-.5);

N3(x,y) :=       if x < 1.0 then 0
            else                 1*(x-1)*(  y);

N4(x,y) :=       if x < 0.5 and y < 0.5 then 4*(.5-x)*(.5-y)
            else                             0;

N5(x,y) :=       if y > 0.5 then 0
            else if x > 1.0 then 0
            else if x < 0.5 then 4*(   x)*(.5-y)
            else                 4*(1.-x)*(.5-y);

N6(x,y) :=       if x < 0.5 and y < 0.5 then 4*(.5-x)*(   y)
            else if x < 0.5 and y > 0.5 then 4*(.5-x)*(1.-y)
            else                             0;

N7(x,y) :=       if x > 1.0             then 0
            else if x < 0.5 and y < 0.5 then 4*(   x)*(   y)
            else if x > 0.5 and y < 0.5 then 4*(1.-x)*(   y)
            else if x < 0.5 and y > 0.5 then 4*(   x)*(1.-y)
            else if x > 0.5 and y > 0.5 then 4*(1.-x)*(1.-y);

N8(x,y) :=       if x > 1.0             then 0
            else if x < 0.5             then 0
            else if x > 0.5 and y < 0.5 then 4*(x-.5)*(   y)
            else                             4*(x-.5)*(1.-y);

N(x,y) := [N0(x,y) + 0.5*N8(x,y), N1(x,y),
           N2(x,y) + 0.5*N8(x,y), N3(x,y),
           N4(x,y),               N5(x,y),
           N6(x,y),               N7(x,y)];

Nk_a : apply(N, [a[1], a[2]]);
Nk_b : apply(N, [b[1], b[2]]);

M : zeromatrix (8, 2);

for k:1 thru 8 do
  (
    M[k][1] : factor * nab[1] * (Nk_a[k] - Nk_b[k]),
    M[k][2] : factor * nab[2] * (Nk_a[k] - Nk_b[k])
  );

float(M);

//------------------------------------------------------------------------------
Ni(x,y) above are all local shape functions and
CNi(x,y) are composite shape functions.
plot3d (N8(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
*/
