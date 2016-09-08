#ifndef __dealii_qc_qc_h
#define __dealii_qc_qc_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

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

    const FEValuesExtractors::Vector u_fe;

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


    /**
     * Unknown displacement field.
     */
    vector_t displacement;

    /**
     * Gradient of the energy (a scalar) w.r.t. to the displacement field.
     */
    vector_t gradient;

    /**
     * Locally relevant displacement filed
     */
    vector_t locally_relevant_displacement;

    /**
     * Auxiliary class with all the information needed per cell for
     * calculation of energy and forces in quasi-continuum method.
     *
     * Since initial positions of atoms is generally random in each
     * element, we have to have a separate FEValues object for each cell.
     *
     * The most tricky part in non-local methods like molecular mechanics
     * within the FE approach is to get the following association link:
     *
     * cell -> atom_id -> neighbour_id -> neighbour_cell -> local_neighbour_id
     *
     */
    struct AssemblyData
    {
      AssemblyData()
      {
      };

      ~AssemblyData()
      {
        Assert(fe_values.use_count() < 2,
               ExcMessage("use count: " + std::to_string(fe_values.use_count())));
      }

      /**
       * FEValues object to evaluate fields and shape function values at
       * quadrature points.
       */
      std::shared_ptr<FEValues<dim>> fe_values;

      /**
       * All atoms attributed to this cell.
       *
       * TOOD: move away from this struct? Do-once-and-forget.
       */
      std::vector<unsigned int> cell_atoms;

      /**
       * A map of global atom IDs to quadrature point (local id of at atom)
       */
      std::map<unsigned int, unsigned int> quadrature_atoms;

      /**
       * IDs of all atoms which are needed for energy calculation.
       */
      std::vector<unsigned int> energy_atoms;

      /**
       * A vector to store displacements evaluated at quadrature points
       */
      mutable std::vector<Tensor<1,dim>> displacements;

      /**
       * A map for each cell to related global degree-of-freedom, to those
       * defined on the cell. Essentially, the reverse of
       * cell->get_dof_indices().
       */
      std::map<unsigned int, unsigned int> global_to_local_dof;

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
