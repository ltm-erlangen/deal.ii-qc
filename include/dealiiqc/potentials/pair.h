

#ifndef __dealii_qc_pair_h
#define __dealii_qc_pair_h

#include <dealiiqc/atom/atom_data.h>
#include <dealiiqc/assembly_data.h>
#include <dealiiqc/io/configure_qc.h>


namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

namespace dealiiqc
{

  /**
   * A namespace to define all the interaction potentials.
   */
  namespace Potential
  {
    /**
     * A enum for keeping a list of all pair potentials implemented in dealiiqc.
     *
     * Should be updated as and when we add more potentials.
     */
    enum PairStyle
    {
      lj_cut = "lj_cut"
    };


    /**
     * A base class for pair potentials.
     */
    template<int dim>
    class Pair
    {
      protected:

        typedef LA::MPI::Vector vector_t;

      public:

        /**
         * Constructor.
         */
        Pair( const ConfigureQC &config);

        /**
         * Virtual destructor.
         */
        virtual ~Pair(){}

        /**
         * Returns the computed value of energy of the atomistic system using
         * QC approach. All derived classes of Pair class must implement this
         * function.
         *
         * During a typical energy minimization process the value function
         * might be called much often than the gradient function. The
         * optimization library ROL uses two separate functions value and
         * gradient for optimizing the objective function.
         * The function call will also update the energy member variables
         * hence not const.
         *
         * The pair potential class is unaware of the kinematics of the problem,
         * it wouldn't by itself update the positions of the locally relevant
         * energy_atoms. This is, perhaps, the work of the core QC class or
         * it's member objects.
         */
        virtual
        double value ( const types::NeighborListsType<dim> &neighbor_lists) = 0;

        /**
         * Updates the gradient vector @p gradient using @p neighbor_lists
         * and @p cell_assembly_data.
         * All derived classes of Pair class must implement this function.
         *
         * The gradient vector represents the the derivative of energy of the
         * atomistic system using QC approach.
         */
        virtual
        void gradient ( vector_t &gradient,
                        const types::CellAssemblyData<dim> &cell_assembly_data,
                        const types::NeighborListsType<dim> &neighbor_lists) const = 0;

        /**
         * Parse and setup pair potential coefficients. The function also
         * upadtes interacting pair of atom types.
         * All derived classes of Pair class must implement this function.
         *
         * The pair potential coefficients could be different for interactions
         * between different atom types.
         */
        virtual
        void parse_pair_coefficients () = 0;

        /**
         * Initialize energy_per_atom
         */
        void initialize_energy_per_cluster_atom ( unsigned int n);

        // TODO: Implement the below function. The function should be
        //       implemented in the base class. There can be other
        //       functions checking pair potential specific legalities.
        /**
         * General sanity checks on the ConfigureQC object @see configure_qc.
         *
         * These sanity checks are necessary but not sufficient to guarentee a
         * legal ConfigureQC object. For example,
         * - Maximum energy radius specified should be above all the cutoff
         * radii as it is used to initialize the neighbor_lists.
         * - If the specified PairStyle is legal.
         */
        void general_checks () const;

      private:

        /**
         * A const reference to the ConfigureQC object to initialize
         * Pair potential attributes.
         */
        const ConfigureQC &configure_qc;

        /**
         * Map of interacting pair of atom types.
         */
        types::InteractingPair interacting_pair;

        /**
         * Per cluster atom energy.
         *
         * The size of the vector is set once after the energy_atoms is updated.
         */
        std::vector<double> energy_per_cluster_atom;

    };

  } // namespace Potential

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_h */
