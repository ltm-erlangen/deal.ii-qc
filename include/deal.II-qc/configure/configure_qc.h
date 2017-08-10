
#ifndef __dealii_qc_configure_qc_h
#define __dealii_qc_configure_qc_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_fire.h>

#include <fstream>
#include <sstream>
#include <utility>
#include <memory>

#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_lumped_vertex.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_sampling_points.h>
#include <deal.II-qc/configure/geometry/geometry_box.h>
#include <deal.II-qc/configure/geometry/geometry_gmsh.h>
#include <deal.II-qc/potentials/pair_coul_wolf.h>
#include <deal.II-qc/potentials/pair_lj_cut.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * A class to read qc input parameter file.
 * The input parameter file should contain the following information:
 * - Initial mesh information
 * - Atoms attributes
 * - Problem dependent operations
 */
class ConfigureQC
{
public:

  /**
   * Parameters to setup SolverControl.
   */
  struct SolverControlParameters
  {
    /**
     * Maximum number of minimizer iterations before declaring failure.
     */
    unsigned int max_steps;

    /**
     * Prescribed tolerance to be achieved.
     */
    double       tolerance;

    /**
     * Log convergence history to deallog.
     */
    bool         log_history;

    /**
     * Log only every nth step.
     */
    unsigned int log_frequency;

    /**
     * If true, after finishing the iteration, a statement about failure or
     * success together with last step and value of convergence criteria are
     * logged.
     */
    bool         log_result;
  };

  /**
   * Parameters to setup SolverFIRE minimizer.
   */
  struct FireParameters
  {
    /**
     * Initial time step.
     */
    double initial_time_step;

    /**
     * Maximum time step.
     */
    double maximum_time_step;

    /**
     * Maximum linfty norm.
     */
    double maximum_linfty_norm;
  };

  /**
   * Parameters for custom initial refinement.
   */
  struct InitialRefinementParameters
  {
    /**
     * Function expression that describes a field to flag
     * cells for refinement. If the value of the function
     * evaluated at cell centers is above a certain threshold,
     * the cell is to be marked for refinement.
     */
    std::string refinement_function;

    /**
     * The fraction of cells to be refined. If this number is zero, no cells
     * will be refined. If it equals one, the result will be flagging for
     * global refinement.
     */
    double fraction_of_cells;

    /**
     * Number of refinement cycles.
     */
    unsigned int n_refinement_cycles;
  };

  /**
   * Constructor with a shared pointer to an istream object @p is.
   */
  ConfigureQC( std::shared_ptr<std::istream> is);

  /**
   * Get dimensionality of the problem
   */
  unsigned int get_dimension() const;

  /**
   * Get pair potential type.
   */
  std::string get_pair_potential_type() const;

  /**
   * Get a shared pointer to const dim dimensional Geometry object.
   *
   * The function returns one of #geometry_3d, #geometry_2d or #geometry_1d
   * depending on the dimension provided in the input script while
   * constructing the ConfigureQC object.
   */
  template<int dim>
  std::shared_ptr<const Geometry::Base<dim>> get_geometry () const;

  /**
   * Get atom data file.
   */
  std::string get_atom_data_file() const;

  /**
   * Get input stream
   */
  std::shared_ptr<std::istream> get_stream() const;

  /**
   * Return #ghost_cell_layer_thickness.
   */
  double get_ghost_cell_layer_thickness() const;

  // TODO: take maximum_cutoff_radius from pair potential cutoff radii?
  // maximum_cutoff_radius= max{ cutoff_radii } + skin?
  /**
   * Return #maximum_cutoff_radius.
   */
  double get_maximum_cutoff_radius() const;

  /**
   * Get cluster radius.
   */
  double get_cluster_radius() const;

  /**
   * Get a shared pointer to the pair potential class object.
   */
  std::shared_ptr<Potential::PairBaseManager> get_potential() const;

  /**
   * Create and return a shared pointer to the derived class object of
   * WeightsByBase.
   */
  template <int dim, int atomicity=1, int spacedim=dim>
  std::shared_ptr<Cluster::WeightsByBase<dim, atomicity, spacedim> >
  get_cluster_weights() const;

  /**
   * Get the map from boundary ids to boundary function expressions in
   * the string format.
   */
  std::map<unsigned int, std::vector<std::string> >
  get_boundary_functions() const;

  /**
   * Get the map describing external potential field expressions in the string
   * format. See #external_potential_field_expressions for explanation of the
   * returned map.
   */
  std::map<std::pair<unsigned int, bool>, std::string>
  get_external_potential_fields() const;

  /**
   * Get minimizer's name.
   */
  std::string get_minimizer_name() const;

  /**
   * Get the time interval between load steps during the quasi-static loading
   * process.
   */
  double get_time_step() const;

  /**
   * Get the number of load steps during the quasi-static loading process.
   */
  unsigned int get_n_time_steps() const;

  /**
   * Get SolverControl parameters.
   */
  SolverControlParameters get_solver_control_parameters () const;

  /**
   * Get SolverFIRE parameters.
   */
  FireParameters get_fire_parameters() const;

  /**
   * Get parameters for custom initial refinement.
   */
  InitialRefinementParameters get_initial_refinement_parameters() const;

private:

  /*
   * Declare parameters to configure QC class.
   */
  static void declare_parameters( ParameterHandler &prm );

  /*
   * Parse parameters
   */
  void parse_parameters( ParameterHandler &prm );

  /**
   * Dimensionality of the problem
   */
  unsigned int dimension;

  /**
   * Pair potential type to be used.
   */
  std::string pair_potential_type;

  /**
   * Shared pointer to the three dimensional Geometry object.
   *
   * @note The dimension of the Geometry object has to be read from the input
   * stream, therefore in ConfigureQC class, shared pointers to all the three
   * dimensioned Geometry object is declared. However, upon reading the input
   * stream only the relevant shared pointer is initialized and the rest of
   * the shared pointers are NULL.
   */
  std::shared_ptr<const Geometry::Base<3>> geometry_3d;

  /**
   * Shared pointer to the two dimensional Geometry object.
   *
   * See note in #geometry_3d
   */
  std::shared_ptr<const Geometry::Base<2>> geometry_2d;

  /**
   * Shared pointer to the one dimensional Geometry object.
   *
   * See note in #geometry_3d
   */
  std::shared_ptr<const Geometry::Base<1>> geometry_1d;

  /**
   * Path to the atom data file.
   */
  std::string atom_data_file;

  /**
   * Shared pointer to the input stream passed in to the
   * constructor ConfigureQC().
   */
  mutable std::shared_ptr<std::istream> input_stream;

protected:

  /**
   * A shared pointer to the pair potential object.
   */
  mutable std::shared_ptr<Potential::PairBaseManager> pair_potential;

  /**
   * Maximum number of boundary ids with specified boundary conditions.
   */
  static const unsigned int max_n_boundaries = 10;

  /**
   * A map from boundary ids to function expressions which describe the boundary
   * conditions.
   */
  std::map<unsigned int, std::vector<std::string> >
  boundary_ids_to_function_expressions;

  /**
   * Maximum number of mateiral ids in the domain.
   */
  static const unsigned int max_n_material_ids = 10;

  /**
   * A map to describe external potential field function expressions.
   * The mapping is from a pair of unsigned int (describing the
   * material id on which the external potential field is prescribed) and
   * bool (whether the external potential field is an electric field) to the
   * external potential field function expression.
   */
  std::map<std::pair<unsigned int, bool>, std::string>
  external_potential_field_expressions;

  /**
   * In distributed memory calculation with local h-adaptive FE each
   * MPI process stores a non-overlapping subset of all cells, such that
   * their union gives the global mesh. Additionally each core stores a
   * layer of so called ghost cells. For non-local methods such as QC,
   * the calculation of energy functional and its derivative may require
   * more than a halo layer around locally owned cells. This is due to the
   * non-local energy evaluation for pair potentials where an energy atom
   * within a locally owned cell may interact with another ghost atom
   * located in a cell not owned by this MPI process.
   *
   * The #ghost_cell_layer_thickness is used to build a layer of ghost cells
   * around locally owned cells such that in the reference configuration all
   * atoms that are no further than #ghost_cell_layer_thickness away from the
   * cluster atoms for a given MPI core are located only within the union of
   * locally owned cells and ghost cells. This algorithm relies on the upper
   * bound estimation of the distance between two arbitrary cells using
   * TriaAccessor::enclosing_ball().
   *
   * @note Note that for physically correct results
   * #ghost_cell_layer_thickness should always be more or equal to the
   * maximum cut-off radius used in a pair potential. Since the ghost cells
   * are determined once at the beginning of calculations, the value of this
   * parameter should be large enough to account for expected deformations in
   * the system so that at each step of applied load the ghost cells contain
   * all ghost atoms.
   */
  double ghost_cell_layer_thickness;

  /**
   * The maximum of all the cutoff radii of the pair potentials.
   * It is used to update neighbor lists of atoms.
   *
   * @note #maximum_cutoff_radius is different from
   * #ghost_cell_layer_thickness which is used to identify locally relevant
   * ghost cells of MPI processes.
   */
  double maximum_cutoff_radius;

  /**
   * The cluster radius used in QC.
   */
  double cluster_radius;

  /**
   * The type of method to update cluster weights.
   */
  std::string cluster_weights_type;

  /**
   * Type of minimizer
   */
  std::string minimizer;

  /**
   * Number of load steps to be performed during the quasi-static loading.
   */
  unsigned int n_time_steps;

  /**
   * The time interval between load steps of quasi-static loading process.
   * The value depends on the sensitivity of the atomistic system to the
   * applied loading.
   */
  double time_step;

  /**
   * SolverControl parameters
   */
  SolverControlParameters solver_control_parameters;

  /**
   * Parameters to setup SolverFIRE minimizer.
   */
  FireParameters fire_parameters;

  /**
   * Initial refinement parameters.
   */
  InitialRefinementParameters initial_refinement_parameters;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif // __dealii_qc_configure_qc_h
