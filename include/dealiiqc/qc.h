#ifndef __dealii_qc_qc_h
#define __dealii_qc_qc_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <dealiiqc/atom/atom.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * Main class for the Quasi-continuum calculations
   */
  template <int dim>
  class QC
  {
  public:
    QC (/*const Parameters<dim> &parameters*/);
    ~QC ();
    void run ();

    // keep it in protected so that we can write unit tests with derived classes
  protected:
    typedef LA::MPI::SparseMatrix matrix_t;
    typedef LA::MPI::Vector vector_t;

    /**
     * Distribute degrees-of-freedom and initialise matrices and vectors.
     */
    void setup_system ();

    /**
     * Main function to calculate radient of the enrgy function
     * (written to @p gradient) and it's value (returned) for a given input
     * @p locally_relevant_displacement.
     */
    double calculate_energy_gradient(const vector_t &locally_relevant_displacement,
                                     vector_t &gradient) const;


    /**
     * Run through all atoms and find a cells to which they belong.
     *
     * TODO: move to a utility function
     */
    void associate_atoms_with_cells();


    /**
     * Given cells and dof handler, for each cell set-up FEValues object with
     * quadrature made of those atoms, which we are interested in. Namely
     * atoms within clusters and also atoms within a cut-off radios of each
     * cluster (one sphere within another).
     *
     * TODO: implement the logic above. For now just use all atoms.
     */
    void setup_fe_values_objects();

    /**
     * MPI communicator
     */
    MPI_Comm mpi_communicator;

    /**
     * Conditional terminal output (root MPI core).
     */
    ConditionalOStream   pcout;

    /**
     * A parallel shared triangulation.
     */
    parallel::shared::Triangulation<dim> triangulation;

    /**
     * Finite Element.
     */
    FESystem<dim>        fe;

    /**
     * Linear mapping.
     */
    MappingQ1<dim>       mapping;

    /**
     * Degrees-of-freedom handler.
     */
    DoFHandler<dim>      dof_handler;

    /**
     * Locally relevant DoFs.
     */
    IndexSet locally_relevant_set;

    /**
     * All constraints (hanging nodes + BC).
     */
    ConstraintMatrix     constraints;


    vector_t displacement;

    vector_t locally_relevant_displacement;

    /**
     * Auxiliary class with all the information needed per cell for
     * calculation of energy and forces in quasi-continuum method.
     *
     * Since ther initial positions of atoms is generally random in each
     * element, we have to have a separate FEValues object for each cell.
     */
    struct AssemblyData
    {
      std::shared_ptr<FEValues<dim>> fe_values;

      /**
       * All atoms attributed to this cell.
       */
      std::vector<unsigned int> all_atoms;

    };

    /**
     * Map of cells to data.
     */
    std::map<typename DoFHandler<dim>::active_cell_iterator, AssemblyData> cells_to_data;

    /**
     * A vector of atoms in the system.
     */
    std::vector<Atom<dim>> atoms;

    /**
     * A time object
     */
    mutable TimerOutput  computing_timer;

  };

}

#endif // __dealii_qc_qc_h
