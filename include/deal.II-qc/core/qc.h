#ifndef __dealii_qc_qc_h
#define __dealii_qc_qc_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/numerics/vector_tools.h>

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


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * A principal class for the fully non-local energy-based quasicontinuum
 * calculations.
 *
 * @note QC only supports quasicontinuum description of a single Molecule
 * type.
 */
template <int dim, typename PotentialType>
class QC
{
public:
  QC (const ConfigureQC &);
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
   * Copy @p configure into #configure_qc member to reconfigure QC without
   * changing CellMoleculeData::cell_molecules of #cell_molecule_data.
   * In doing so the initially set association between the mesh and all the
   * atoms is kept as is; energy molecules should be updated according to the
   * new sampling rules and/or certain parameters provided in @p configure.
   *
   * Typically, parsing atom data and assigning cells to molecules for a given
   * large atomistic system is one of the most time consuming part of a large
   * QC simulation. For comparison of QC simulations with the same initially
   * given atomistic system and the same #triangulation but slightly altered
   * #configure_qc (say because of a slightly different
   * ConfigureQC::cluster_radius), parsing atoms and assigning cells to the
   * atoms again are redundant operations (unless
   * ConfigureQC::ghost_cell_layer_thickness is changed in which the cells
   * and molecules association should be rebuilt). Therefore, to avoided
   * re-setting up of CellMoleculeData::cell_molecules of #cell_molecule_data
   * the current QC object can be reconfigured using this function.
   */
  void reconfigure_qc(const ConfigureQC &configure);

  /**
   * Setup triangulation.
   */
  void setup_triangulation();

  /**
   * Setup few data members in #cell_molecule_data (namely:
   * CellMoleculeData::cell_molecules,
   * CellMoleculeData::masses and
   * CellMoleculeData::charges) of the current MPI process. However, this
   * function doesn't update CellMoleculeData::cell_energy_molecules of
   * #cell_molecule_data, this is done by
   * QC::setup_cell_energy_molecules().
   *
   * @note The primary data member CellMoleculeData::cell_molecules in
   * #cell_molecule_data should be used only to initialize
   * CellMoleculeData::cell_energy_molecules in #cell_molecule_data. And
   * CellMoleculeData::cell_energy_molecules in #cell_molecule_data should be
   * used to compute energy or force using the quasicontinuum approach.
   */
  void setup_cell_molecules();

  /**
   * Setup CellMoleculeData::cell_energy_molecules of #cell_molecule_data of
   * the current MPI process with appropriate cluster weights.
   *
   * All the cluster molecules get a non-zero cluster weight while all the
   * other energy (non-cluster) molecules get a zero cluster weight. Data
   * member #configure_qc creates a shared pointer to a suitable derived
   * class of Cluster::WeightsByBase based on the chosen method
   * (or sampling rule) to update cluster weights of energy molecules.
   */
  void setup_cell_energy_molecules();

  /**
   * Initialize #dirichlet_boundary_functions.
   */
  void initialize_boundary_functions();

  /**
   * Insert the (algebraic) constraints due to Dirichlet boundary conditions
   * into #constraints.
   */
  void setup_boundary_conditions(const double time = 0.);

  /**
   * Distribute degrees-of-freedom and initialise matrices and vectors.
   */
  void setup_system ();

  /**
   * Update positions of the atoms of energy molecules
   * (CellMoleculeData::cell_energy molecules) according to the given
   * @p locally_relevant_displacement of finite element displacement field.
   *
   * @note: During a typical minimization call in ROL, this function is called
   * after each iterate is formed. Internally in ROL, update() function of the
   * objective class is called after each iterate is formed and is responsible
   * for updating the data members of the objective class. Therefore,
   * the current function is separated from compute().
   */
  void update_positions();

  /**
   * Return the computed energy of the atomistic system using QC approach, and
   * update the its @p gradient if <tt>ComputeGradient</tt> is set true.
   *
   * The template parameter indicates whether to do the additional
   * computation of the gradient of the energy; when <tt>ComputeGradient</tt>
   * is set false only the value of the energy is computed.
   */
  template<bool ComputeGradient=true>
  double compute (vector_t &gradient) const;

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
   * Locally relevant unknown displacement filed
   */
  vector_t locally_relevant_displacement;

  /**
   * Gradient of the energy (a scalar) w.r.t. to the displacement field.
   */
  vector_t locally_relevant_gradient;

  /**
   * Map of boundary ids to Functions describing the corresponding boundary
   * condition.
   */
  std::map<unsigned int, std::pair<ComponentMask, std::shared_ptr<FunctionParser<dim> > > >
  dirichlet_boundary_functions;

  /**
   * Auxiliary class with all the information needed per cell for
   * calculation of energy and forces in quasi-continuum method.
   *
   * Since initial positions of molecules is generally random in each
   * element, we have to have a separate FEValues object for each cell.
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

    // FIXME: can we avoid using FEValues completely and do things manually
    // like in Aspect's World<dim>::local_advect_particles()
    // https://github.com/geodynamics/aspect/blob/master/source/particle/world.cc#L1265
    // Aspect's developers saw x2 speedup for their particles in this step
    /**
     * FEValues object to evaluate fields and shape function values at
     * quadrature points.
     *
     * The CellMoleculeData::cell_energy_molecules of #cell_molecule_data holds
     * the association between all the locally relevant active cells and the
     * locally relevant energy molecules on the current MPI process.
     * The positions of molecules' associated to a particular active cell are
     * updated according to the (linearly varying) displacement field within
     * the cell. The displacement within the cell can be obtained using
     * FEValues object of the cell.
     */
    std::shared_ptr<FEValues<dim>> fe_values;

    // TODO: do we really need this? FEValues do store displacement already
    // after calling FEValues::reinit(cell), so we might just ask it directly
    // for those values. It would probably even store it more efficiently
    // internally given the tensor product property of FE basis.
    /**
     * A vector to store displacements evaluated at quadrature points.
     *
     * The size of this vector is exactly equal to the number of energy
     * molecules on a per cell basis. The order of this list is the same order
     * in which the energy molecules are stored in energy_atoms on a per cell
     * basis.
     */
    mutable std::vector<Tensor<1,dim>> displacements;

  };

  /**
   * Map of cells to data.
   */
  std::map<types::DoFCellIteratorType<dim>, AssemblyData>
  cells_to_data;

  /**
   * Shared pointer to the cluster weights method.
   */
  std::shared_ptr<Cluster::WeightsByBase<dim> > cluster_weights_method;

  /**
   * The primary atom data object that holds cell based atom data structures.
   * Cell based atom data structures rely on the association between molecules
   * and mesh.
   */
  CellMoleculeData<dim> cell_molecule_data;

  /**
   * MoleculeHandler object to manage the cell based neighbor lists of the
   * system.
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


DEAL_II_QC_NAMESPACE_CLOSE

#endif // __dealii_qc_qc_h
