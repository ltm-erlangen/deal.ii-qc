
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/core/qc.h>

#include <iomanip>
#include <string>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;


// Check the correctness of the interpretation of DoFs renumbering in
// QC::setup_system().
//
// This test writes out the DoF, component, block and non-zero component
// indices on a single cell in two dimensions.
//
// This test makes sure that certain assumptions made in QC::compute() and
// WeightsByBase::compute_inverse_masses() about DoF numbering hold true.
// If this test fails then something has been changed regarding how DoF
// enumeration/renumbering is performed.
//
// x-------x
// |       |      x  - vertices
// |       |
// |       |      dim       = 2
// x-------x      atomicity = 3
//


template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &config)
    : QC<dim, PotentialType, atomicity>(config)
  {
    this->setup_cell_energy_molecules();
    this->setup_system();
  }

  void
  test();
  void
  write_out();
};


template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::test()
{
  AssertThrow(this->triangulation.n_active_cells() == 1, ExcNotImplemented());

  const unsigned int n_dofs = this->dof_handler.n_dofs();

  std::cout
    << " * | Local DoF | Gloabl DoF | Component | Block (atom stamp) | Non-zero Component |"
    << std::endl
    << " * | :-------: | :--------: | :-------: | :----------------: | :----------------: |"
    << std::endl;

  std::vector<dealii::types::global_dof_index> local_to_global_dof_indices(
    n_dofs);

  this->dof_handler.begin_active()->get_dof_indices(
    local_to_global_dof_indices);

  // Loop over all elements in dof handler.
  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const unsigned int component =
        this->fe.system_to_component_index(i).first;
      const unsigned int atom_stamp   = std::div(component, dim).quot;
      const unsigned int nonzero_comp = component % dim;

      std::cout << std::setfill(' ') << " * | " << std::setw(9) << i << " | "
                << std::setw(10) << local_to_global_dof_indices[i] << " | "
                << std::setw(9) << component << " | " << std::setw(18)
                << atom_stamp << " | " << std::setw(18) << nonzero_comp << " |"
                << std::endl;
    }
  std::cout << std::endl;

  // print grid and DoFs for visual inspection
  if (true)
    {
      std::map<dealii::types::global_dof_index, Point<dim>> support_points;
      MappingQ1<dim>                                        mapping;
      DoFTools::map_dofs_to_support_points(mapping,
                                           this->dof_handler,
                                           support_points);

      const std::string filename =
        "grid" + dealii::Utilities::int_to_string(dim) + ".gp";
      std::ofstream f(filename.c_str());

      f << "set terminal png size 420,440 enhanced font \"Helvetica,16\""
        << std::endl
        << "set output \"grid" << dealii::Utilities::int_to_string(dim)
        << ".png\"" << std::endl
        << "set size square" << std::endl
        << "set view equal xy" << std::endl
        << "unset xtics" << std::endl
        << "unset ytics" << std::endl
        << "unset border" << std::endl
        << "set xrange [0: 1.05]" << std::endl
        << "set yrange [0: 1.05]" << std::endl
        << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 0.5,0.5 notitle"
        << std::endl;
      GridOut().write_gnuplot(this->triangulation, f);
      f << "e" << std::endl;

      DoFTools::write_gnuplot_dof_support_point_info(f, support_points);
      f << "e" << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 2;

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
          << "  set Pair specific coefficients = 1, 1, 0.877, 1.2;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 0.2" << std::endl
          << "  set Cluster weights by type = Cell" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "12 atoms" << std::endl
          << std::endl
          << "3  atom types" << std::endl
          << std::endl
          << "Masses" << std::endl
          << std::endl
          << "    1   0.7" << std::endl
          << std::endl
          << "    2   0.23" << std::endl
          << std::endl
          << "    3   0.43" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl

          << "1   1  1 1.0   0.0 0.0 0." << std::endl
          << "2   1  2 1.0   0.1 0.0 0." << std::endl
          << "3   1  3 1.0   0.0 0.1 0." << std::endl

          << "4   2  1 1.0   1.0 0.0 0." << std::endl
          << "5   2  2 1.0   0.9 0.0 0." << std::endl
          << "6   2  3 1.0   1.0 0.1 0." << std::endl

          << "7   3  1 1.0   0.0 1.0 0." << std::endl
          << "8   3  2 1.0   0.1 1.0 0." << std::endl
          << "9   3  3 1.0   0.0 0.9 0." << std::endl

          << "10  4  1 1.0   1.0 1.0 0." << std::endl
          << "11  4  2 1.0   0.9 1.0 0." << std::endl
          << "12  4  3 1.0   1.0 0.9 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      Problem<dim, Potential::PairLJCutManager, 3> problem(config);
      problem.test();
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
