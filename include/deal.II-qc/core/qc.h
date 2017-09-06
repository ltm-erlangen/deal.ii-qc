#ifndef __dealii_qc_qc_h
#define __dealii_qc_qc_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/config.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/numerics/data_out.h>
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
#include <deal.II-qc/grid/shared_tria.h>
#include <deal.II-qc/potentials/potential_field_function_parser.h>

#include <deal.II-qc/adaptors/rol_vector_adaptor.h>
#ifdef DEAL_II_WITH_TRILINOS
# include <ROL_Objective.hpp>
#endif


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * A principal class for the fully non-local energy-based quasicontinuum
 * calculations.
 *
 * The atomistic system is considered to be consisting of molecules
 * (described using Molecule class) consisting of a fixed
 * <tt>atomicity</tt>-number of atoms. Each atom of a molecule is enumerated
 * with an "atom stamp". To account for the degrees-of-freedom (DoFs)
 * associated with each atom stamp, a displacement field can be defined over
 * the atomistic domain.
 *
 * The DoF indices for a single cell in two-dimensions using the current
 * finite element (#fe) i.e.,
 * @code
 *   FESytem<dim> fe (FE_Q<dim>(1), dim*atomicity)
 *   dof_handler.distribute_dofs (fe);
 * @endcode
 * with atomicity=3 yields the following enumeration:
 *
 * @image html dim2_atomicity3_before_renumbering.png DoF indices before renumbering
 *
 * The displacement field for each atom stamp is encapsulated in
 * the BlockVector #distributed_displacement such that
 * `distributed_displacement.block(i)` yields the displacement field
 * of the \f$ i^{th} \f$ atom stamp. To this end DoF indices need to be
 * appropriately renumbered and classified into different blocks where the
 * number of blocks is equal to the <tt>atomicity</tt>. This can be done by
 * the following code:
 *
 * @code
 *   std::vector<unsigned int> component_to_block_indices (dim*atomicity, 0);
 *   for (int i = 0; i < dim*atomicity; ++i)
 *     component_to_block_indices[i] = std::div(i, dim).quot;
 *   // Renumber DoFs block-wise.
 *   DoFRenumbering::component_wise (dof_handler, component_to_block_indices);
 * @endcode
 *
 * DoF renumbering yields the following DoF enumeration:
 *
 * @image html dim2_atomicity3_after_renumbering.png DoF indices after renumbering
 *
 * After DoF renumbering, DoF indices can be associated to atom stamp,
 * FESystem component and the non-zero component index.
 * For a given local DoF index "i" of a finite element and different indices
 * associated to it can be obtained by the following code:
 * @code
 *   const unsigned int component    = fe.system_to_component_index(i).first;
 *   const unsigned int atom_stamp   = std::div(component, dim).quot;
 *   const unsigned int nonzero_comp = component % dim;
 * @endcode
 *
 * Different indices after DoF renumbering is summarized in the following table:
 *
 * | Local DoF | Gloabl DoF | Component | Block (atom stamp) | Non-zero Component |
 * | :-------: | :--------: | :-------: | :----------------: | :----------------: |
 * |         0 |          0 |         0 |                  0 |                  0 |
 * |         1 |          1 |         1 |                  0 |                  1 |
 * |         2 |          8 |         2 |                  1 |                  0 |
 * |         3 |          9 |         3 |                  1 |                  1 |
 * |         4 |         16 |         4 |                  2 |                  0 |
 * |         5 |         17 |         5 |                  2 |                  1 |
 * |         6 |          2 |         0 |                  0 |                  0 |
 * |         7 |          3 |         1 |                  0 |                  1 |
 * |         8 |         10 |         2 |                  1 |                  0 |
 * |         9 |         11 |         3 |                  1 |                  1 |
 * |        10 |         18 |         4 |                  2 |                  0 |
 * |        11 |         19 |         5 |                  2 |                  1 |
 * |        12 |          4 |         0 |                  0 |                  0 |
 * |        13 |          5 |         1 |                  0 |                  1 |
 * |        14 |         12 |         2 |                  1 |                  0 |
 * |        15 |         13 |         3 |                  1 |                  1 |
 * |        16 |         20 |         4 |                  2 |                  0 |
 * |        17 |         21 |         5 |                  2 |                  1 |
 * |        18 |          6 |         0 |                  0 |                  0 |
 * |        19 |          7 |         1 |                  0 |                  1 |
 * |        20 |         14 |         2 |                  1 |                  0 |
 * |        21 |         15 |         3 |                  1 |                  1 |
 * |        22 |         22 |         4 |                  2 |                  0 |
 * |        23 |         23 |         5 |                  2 |                  1 |
 *
 *
 *
 * @note QC only supports quasicontinuum description of a single Molecule
 * type.
 */
template <int dim, typename PotentialType, int atomicity=1>
class QC
{
  // TODO: Remove this after adding spacedim as template parameter.
  static const unsigned int spacedim = dim;
public:

  typedef LA::MPI::BlockVector vector_t;

  /**
   * Constructor.
   */
  QC (const ConfigureQC &);

  virtual ~QC ();

  /**
   * Perform QC simulation based on #configure_qc. If
   * @p relaxed_configuration_as_reference is true then the displacement is
   * measured from the relaxed configuration i.e., after an energy minimization
   * step without applying any external potential. Otherwise, the displacement
   * is measured from the initial configuration.
   */
  void run (const bool relaxed_configuration_as_reference=true);

#ifdef DEAL_II_WITH_TRILINOS
  /**
   * Class for defining ROL library compliant objective function using the
   * current QC object.
   *
   * After QC::distributed_displacement is prepared with locally owned
   * DoF indices, it can be used to prepare an rol::VectorAdaptor of
   * VectorType QC::vector_t.
   *
   * The following code illustrates how ROL library can be used.
   *
   * @code
   *   Teuchos::RCP<vector_t> x_rcp = Teuchos::rcp (&distributed_displacement);
   *   rol::VectorAdaptor<vector_t> x(x_rcp);
   * @endcode
   *
   * The parameter list for choosing a particular minimization scheme can be
   * prepared as
   *
   * @code
   *   Teuchos::ParameterList parlist;
   *   parlist.sublist("Secant").set("Type", "Limited-Memory BFGS");
   * @endcode
   *
   * or using an xml file of parameters as
   *
   * @code
   *   std::string filename = "rol_input.xml";
   *   Teuchos::RCP<Teuchos::ParameterList> parlist = Teuchos::rcp (new Teuchos::ParameterList());
   *   Teuchos::updateParametersFromXmlFile (filename, parlist.ptr());
   * @endcode
   *
   * For logging per iteration information,
   * @code
   *   Teuchos::RCP<std::ostream> out_stream;
   *   Teuchos::oblackholestream bhs; // outputs nothing
   *
   *   if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
   *     out_stream = Teuchos::rcp(&std::cout, false);
   *   else
   *     out_stream = Teuchos::rcp(&bhs, false);
   *
   *  // Run Algorithm
   *  ROL::Algorithm<double> algorithm ("Line Search", *parlist);
   *  algorithm.run(x, qc_objective, true, *out_stream);
   * @endcode
   *
   */
  class Objective : public ROL::Objective<double>
  {
  public:

    /**
     * Constructor.
     */
    Objective (QC &qc)
      :
      qc(qc),
      energy(0.)
    {}

    Teuchos::RCP<const vector_t>
    get_rcp_to_vector (const ROL::Vector<double> &x)
    {
      return
        Teuchos::dyn_cast<const rol::VectorAdaptor<vector_t> >(x).getVector();
    }

    Teuchos::RCP<vector_t>
    get_rcp_to_vector (ROL::Vector<double> &x)
    {
      return
        Teuchos::dyn_cast<rol::VectorAdaptor<vector_t> >(x).getVector();
    }

    double value (const ROL::Vector<double> &/* x */,
                  double                    &/* tol */)
    {
      return energy;
    }

    void gradient (ROL::Vector<double>       &g,
                   const ROL::Vector<double> &/* x */,
                   double                    &/* tol */)
    {
      *get_rcp_to_vector(g) = qc.locally_relevant_gradient;
    }

    void update (const ROL::Vector<double> &x, bool flag, int iter)
    {
      if (!flag)
        return;

      (void) iter;

      qc.distributed_displacement = *get_rcp_to_vector(x);
      qc.constraints.distribute(qc.distributed_displacement);
      qc.locally_relevant_displacement = qc.distributed_displacement;
      qc.update_positions();
      qc.update_neighbor_lists();
      energy = qc.compute<true>(qc.locally_relevant_gradient);
    }

  private:
    /**
     * A reference to the current QC object.
     */
    QC &qc;

    /**
     * Energy of the atomistic system computed using QC approach.
     */
    double energy;

  };

  /**
   * ROL library compliant objective for the current QC object.
   */
  Objective qc_objective;
#endif

  /**
   * Write mesh file into some sort of ostream object
   * (passed as the first argument).
   * The type of file should be passed as second argument (eps, msh etc)
   */
  template<typename T>
  void write_mesh(T &, const std::string &);

  /**
   * Set up external potential fields.
   */
  virtual
  void initialize_external_potential_fields (const double initial_time = 0.);

  // keep it in protected so that we can write unit tests with derived classes
protected:

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
   * update its @p gradient if <tt>ComputeGradient</tt> is set true.
   *
   * The template parameter indicates whether to do the additional
   * computation of the gradient of the energy; when <tt>ComputeGradient</tt>
   * is set false only the value of the energy is computed.
   */
  template<bool ComputeGradient=true>
  double compute (vector_t &gradient) const;

  /**
   * Return the computed energy of the atomistic system using QC approach, and
   * update @p gradient of the energy upon applying a given @p displacement.
   */
  double compute (vector_t       &gradients,
                  const vector_t &displacements);

  /**
   * Minimize the energy (computed using the QC approach) of the atomistic
   * system at time @p time.
   */
  void minimize_energy (const double time);

  /**
   * Output displacement field at @p time time occurring at time step number
   * @p timestep_no.
   */
  void output_results (const double time,
                       const unsigned int timestep_no) const;

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
  MPI_Comm                         mpi_communicator;

  /**
   * Conditional terminal output (root MPI core).
   */
  ConditionalOStream               pcout;

  /**
   * Read input filename and configure mesh, atoms, etc
   */
  ConfigureQC                      configure_qc;

  /**
   * A parallel shared triangulation.
   */
  dealiiqc::parallel::shared::Triangulation<dim, spacedim> triangulation;

  /**
   * Finite Element.
   */
  FESystem<dim, spacedim>          fe;

  /**
   * Exctractors for displacement fields.
   */
  std::array<FEValuesExtractors::Vector, atomicity> u_fe;

  /**
   * Linear mapping.
   */
  MappingQ1<dim, spacedim>         mapping;

  /**
   * Degrees-of-freedom handler.
   */
  DoFHandler<dim, spacedim>        dof_handler;

  /**
   * All constraints (hanging nodes + BC).
   */
  ConstraintMatrix                 constraints;

  /**
   * Distributed displacement field.
   */
  vector_t                         distributed_displacement;

  /**
   * Locally relevant unknown displacement filed
   */
  vector_t                         locally_relevant_displacement;

  /**
   * Gradient of the energy (a scalar) w.r.t. to the displacement field.
   */
  vector_t                         locally_relevant_gradient;

  /**
   * Inverse mass matrix.
   */
  DiagonalMatrix<vector_t>         inverse_mass_matrix;

  /**
   * Map of boundary ids to Functions describing the corresponding boundary
   * condition.
   */
  std::map<unsigned int, std::pair<ComponentMask, std::shared_ptr<FunctionParser<spacedim> > > >
  dirichlet_boundary_functions;

  /**
   * External potential field function.
   */
  std::multimap<unsigned int, std::shared_ptr<PotentialField<spacedim> > >
  external_potential_fields;

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
    std::shared_ptr<FEValues<dim, spacedim>> fe_values;

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
    mutable std::array<std::vector<Tensor<1,dim>>, atomicity> displacements;

  };

  /**
   * Map of cells to data.
   */
  std::map<types::DoFCellIteratorType<dim, spacedim>, AssemblyData>
  cells_to_data;

  /**
   * Shared pointer to the cluster weights method.
   */
  std::shared_ptr<Cluster::WeightsByBase<dim, atomicity, spacedim> > cluster_weights_method;

  /**
   * The primary atom data object that holds cell based atom data structures.
   * Cell based atom data structures rely on the association between molecules
   * and mesh.
   */
  CellMoleculeData<dim, atomicity, spacedim>                 cell_molecule_data;

  /**
   * MoleculeHandler object to manage the cell based neighbor lists of the
   * system.
   */
  MoleculeHandler<dim, atomicity, spacedim>                  molecule_handler;

  /**
   * Neighbor lists using cell approach.
   */
  types::CellMoleculeNeighborLists<dim, atomicity, spacedim> neighbor_lists;

  /**
   * A time object
   */
  mutable TimerOutput                                        computing_timer;

private:

  /**
   * Return the computed energy of the atomistic system using QC approach
   * for the current MPI process, and update its @p gradient for the current
   * MPI process if <tt>ComputeGradient</tt> is set true.
   */
  template <bool ComputeGradient=true>
  double compute_local (vector_t &gradient) const;

};


DEAL_II_QC_NAMESPACE_CLOSE

#endif // __dealii_qc_qc_h
