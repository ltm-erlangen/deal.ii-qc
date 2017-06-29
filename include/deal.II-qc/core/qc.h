#ifndef __dealii_qc_qc_h
#define __dealii_qc_qc_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

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

#include <deal.II-qc/atom/molecule_handler.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/potentials/pair_lj_cut.h>
#include <deal.II-qc/potentials/pair_coul_wolf.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * Main class for the Quasi-continuum calculations
   */
  template <int dim, typename PotentialType>
  class QC
  {
  public:
    QC ( const ConfigureQC &);
    ~QC ();

    void run ();

    /**
     * Write mesh file into some sort of ostream object
     * (passed as the first argument).
     * The type of file should be passed as second argument (eps, msh etc)
     */
    template<typename T>
    void write_mesh(T &, const std::string &);

    // keep it in protected so that we can write unit tests with derived classes
  protected:
    typedef LA::MPI::SparseMatrix matrix_t;
    typedef LA::MPI::Vector vector_t;

    /**
     * Copy @p configure into #configure_qc member reconfigure QC without
     * changing #atom_data.atoms.
     * In doing so the association between the mesh and all the atoms is kept
     * as is; energy atoms can be updated according to the new sampling rules
     * and parameters given in @p configure.
     *
     * Typically, parsing atoms and assigning cells to the atoms for a given
     * large atomistic system is one of the most time consuming part of a large
     * QC simulation. For comparison of QC simulations with the same initially
     * given atomistic system and the same #triangulation but slightly altered
     * #configure_qc (say because of a slightly different
     * ConfigureQC::cluster_radius or ConfigureQC::ghost_cell_layer_thickness),
     * parsing atoms and assigning cells to the atoms again are redundant
     * operations. Therefore, to avoided re-setting up of #atom_data.atoms
     * the current QC object can be reconfiguring QC using this function.
     */
    void reconfigure_qc(const ConfigureQC &configure);

    /**
     * Setup triangulation.
     */
    void setup_triangulation();

    /**
     * Setup few data members in #atom_data (namely: AtomData::atoms,
     * AtomData::masses and AtomData::atomtype_to_atoms) of the current
     * MPI process. However, this function doesn't update
     * AtomData::energy_atoms in #atom_data, this is done by
     * QC::setup_energy_atoms_with_cluster_weights().
     *
     * This function initializes the primary data member AtomData::atoms
     * which holds an association between the locally relevant active cells of
     * the underlying #triangulation and all the atoms in it. This is
     * accomplished by calling AtomHandler::parse_atoms_and_assign_to_cells().
     *
     * @note The primary data member AtomData::atoms in #atom_data should be
     * used only to initialize AtomData::energy_atoms in #atom_data. And
     * AtomData::energy_atoms in #atom_data should be used to compute energy or
     * force using the quasicontinuum approach.
     */
    void setup_atoms();

    /**
     * Setup AtomData::energy_atoms, with appropriate cluster weights,
     * in #atom_data of the current MPI process.
     *
     * All the cluster atoms get a non zero cluster weight while the
     * non-cluster atoms get a zero cluster weight. #configure_qc creates a
     * shared pointer to the derived class of Cluster::WeightsByBase based on
     * the chosen method (or sampling rule) to update cluster weights of
     * cluster atoms.
     */
    void setup_energy_atoms_with_cluster_weights();

    /**
     * Distribute degrees-of-freedom and initialise matrices and vectors.
     */
    void setup_system ();

    /**
     * Update positions of the energy atoms according to the given
     * @p locally_relevant_displacement of finite element displacement field.
     *
     * @note: During a typical minimization call in ROL, this function is called
     * after each iterate is formed. Internally in ROL, update() function of the
     * objective class is called after each iterate is formed and is responsible
     * for updating the data members of the objective class. Therefore,
     * update_energy_atoms_positions() is separated from
     * calculate_energy_gradient().
     */
    void update_energy_atoms_positions();

    /**
     * Return the computed energy of the atomistic system using QC approach, and
     * update the its @p gradient if @tparam ComputeGradient is set true.
     *
     * The template parameter indicates whether to do the additional
     * computation of the gradient of the energy; when @tparam ComputeGradient
     * is set false only the value of the energy is computed.
     */
    template<bool ComputeGradient=true>
    double calculate_energy_gradient (vector_t &gradient) const;

    // TODO: implement the logic above. For now just use all atoms.
    /**
     * Given cells and dof handler, for each cell set-up FEValues object with
     * quadrature made of those atoms, which we are interested in. Namely
     * atoms within clusters and also atoms within a cut-off radios of each
     * cluster (one sphere within another).
     */
    void setup_fe_values_objects();

    /**
     * Update neighbor lists.
     */
    void update_neighbor_lists();

    /**
     * MPI communicator
     */
    MPI_Comm mpi_communicator;

    /**
     * Conditional terminal output (root MPI core).
     */
    ConditionalOStream   pcout;

    /**
     * Read input filename and configure mesh, atoms, etc
     */
    ConfigureQC configure_qc;

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
       *
       * The @see energy_atoms data member of AtomData holds association between
       * all the locally relevant active cells and the locally relevant energy
       * atoms of the current MPI process. The positions of atoms' associated
       * to a particular active cell are updated according to the (linearly
       * varying) displacement field within the cell.
       * The displacement within the cell can be obtained using FEValues object
       * of the cell.
       */
      std::shared_ptr<FEValues<dim>> fe_values;

      // TODO: do we really need this? FEValues do store displacement already
      // after calling FEValues::reinit(cell), so we might just ask it directly
      // for those values. It would probably even store it more efficiently
      // internally given the tensor product property of FE basis.
      /**
       * A vector to store displacements evaluated at quadrature points.
       *
       * The size of this vector is exactly equal to the number of energy_atoms
       * on a per cell basis. The order of this list is the same order in which
       * the energy atoms are stored in energy_atoms on a per cell basis.
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
     * The primary atom data object that holds cell based atom data structures.
     * Cell based atom data structures rely on the association between atoms
     * and mesh.
     */
    CellMoleculeData<dim> cell_molecule_data;

    /**
     * AtomHandler object to manage the cell based atom data mainly through
     * initializing or updating atom_data.
     */
    MoleculeHandler<dim> molecule_handler;

    /**
     * Neighbor lists using cell approach.
     */
    types::CellMoleculeNeighborLists<dim> neighbor_lists;

    /**
     * A time object
     */
    mutable TimerOutput  computing_timer;

  };


} // namespace dealiiqc

#endif // __dealii_qc_qc_h
