#ifndef __dealii_qc_minimizer_fire_h
#define __dealii_qc_minimizer_fire_h

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver.h>

#include <deal.II-qc/utilities.h>

#include <functional>


DEAL_II_QC_NAMESPACE_OPEN


namespace statics
{

  using namespace dealii;


  // TODO: Complete the description.
  /**
   * A class implementation of FIRE (Fast Inertial Relaxation Engine) algorithm
   * which is a damped dynamics method described in <a href="">Structural
   * Relaxation Made Simple</a> by Bitzek et al. 2006 and <a href="">
   * Energy-Minimization in Atomic-to-Continuum Scale-Bridging Methods
   * </a> by Eidel et al. 2011, explained below.
   *
   * The strategy to descent to a minimum of the total energy is to follow an
   * equation of motion given by:
   * \f[
   *      \dot{\textbf{V}_i}(t) = -\textbf{G}_i(t)/m_i
   *                            - \gamma |\textbf{V}_i|
   *                             \left[  \hat{\textbf{V}}_i(t) +
   *                                     \hat{\textbf{G}}_i(t)
   *                             \right] \,\,
   *                             \forall \,\, i \in \mathcal D.
   * \f]
   */
  template<typename VectorType>
  class SolverFIRE : public Solver<VectorType>
  {

  public:

    struct AdditionalData
    {
      explicit
      AdditionalData (const double  timestep     = 1e-12,
                      const double  max_timestep = 1e-9,
                      const double  dmax         = 1e-10,
                      std::shared_ptr<const typename dealii::DiagonalMatrix<VectorType>> inverse_masses = nullptr);

      /**
       * Time step.
       */
      const double timestep;

      /**
       * Maximum timestep.
       */
      const double max_timestep;

      /**
       * Maximum change allowed in any degree of freedom.
       */
      const double dmax;

      /**
       * A const reference to the masses in the system.
       */
      const std::shared_ptr<const dealii::DiagonalMatrix<VectorType>> inverse_masses;

    };

    /**
     * Constructor.
     */
    SolverFIRE (SolverControl            &solver_control,
                VectorMemory<VectorType> &vector_memory,
                const AdditionalData     &data          );

    SolverFIRE (SolverControl         &solver_control,
                const AdditionalData  &data          );

    /**
     * Virtual destructor.
     */
    virtual ~SolverFIRE();

    /**
     * Obtain a set of #p u (variables) that minimize an objective function
     * described by the polymorphic function wrapper @p compute. The function
     * @p compute takes in the state of the (u) variables as argument and
     * returns a pair of objective function's value and objective function's
     * gradient (with respect to the variables).
     */
    void solve
    (std::function<double(VectorType &, const VectorType &)>  compute,
     VectorType                                              &u      );

  protected:

    /**
     * Interface for derived class. This function gets the current iteration
     * u, u's time derivative and the gradient in each step. It can be used
     * for a graphical output of the convergence history.
     */
    void print_vectors (const unsigned int,
                        const VectorType &,
                        const VectorType &,
                        const VectorType &) const;

    /**
     * Additional parameters.
     */
    const AdditionalData additional_data;

  };

  /* --------------------- Inline and template functions ------------------- */
#ifndef DOXYGEN

  template<typename VectorType>
  SolverFIRE<VectorType>::AdditionalData::
  AdditionalData (const double  timestep,
                  const double  max_timestep,
                  const double  dmax,
                  std::shared_ptr<const typename dealii::DiagonalMatrix<VectorType>> inverse_masses)
    :
    timestep(timestep),
    max_timestep(max_timestep),
    dmax(dmax),
    inverse_masses(inverse_masses)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::SolverFIRE (SolverControl            &solver_control,
                                      VectorMemory<VectorType> &vector_memory,
                                      const AdditionalData             &data  )
    :
    Solver<VectorType>(solver_control, vector_memory),
    additional_data(data)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::SolverFIRE (SolverControl         &solver_control,
                                      const AdditionalData  &data          )
    :
    Solver<VectorType>(solver_control),
    additional_data(data)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::~SolverFIRE()
  {}



  template<typename VectorType>
  void
  SolverFIRE<VectorType>::solve
  (std::function<double(VectorType &, const VectorType &)>  compute,
   VectorType                                              &u      )
  {
    // FIRE algorithm constants
    const double DELAYSTEP       = 5;
    const double TIMESTEP_GROW   = 1.1;
    const double TIMESTEP_SHRINK = 0.5;
    const double ALPHA_0         = 0.1;
    const double ALPHA_SHRINK    = 0.99;

    using real_type = typename VectorType::real_type;

    VectorType *v, *g;

    v = this->memory.alloc();
    g = this->memory.alloc();

    // Refer to v and g with some readable names.
    VectorType &velocities = *v;
    VectorType &gradients  = *g;

    // Set velocities to zero.
    velocities.reinit(u, false);

    // Don't set Gradients to zero.
    gradients.reinit(u, true);

    // Update gradients for the new u.
    compute(gradients, u);

    SolverControl::State conv = SolverControl::iterate;
    conv = this->iteration_status (0, gradients * gradients, u);
    if (conv != SolverControl::iterate)
      return;

    // Refer to additional data members with some readable names.
    const auto &inverse_masses = additional_data.inverse_masses;
    const auto &max_timestep   = additional_data.max_timestep;
    double timestep       = additional_data.timestep;

    try
      {
        // First scaling factor.
        double alpha = ALPHA_0;

        unsigned int previous_iter_with_positive_v_dot_g = 0;
        double minimal_timestep = timestep;

        unsigned int iter = 0;

        while (conv == SolverControl::iterate)
          {
            // Euler integration step.
            u.sadd (minimal_timestep, velocities);         // U  = dt * V
            inverse_masses->vmult(gradients, gradients);   // G  =      G / M
            velocities.sadd(-minimal_timestep, gradients); // V -= dt * G

            gradients = 0.; //FIXME: QC::compute() should probably also do this?

            // Update gradients for the new u.
            compute(gradients, u);

            const real_type gradient_norm_squared = gradients * gradients;
            conv = this->iteration_status(iter, gradient_norm_squared, u);
            if (conv != SolverControl::iterate)
              break;

            // v_dot_g = V * G
            const real_type v_dot_g = velocities * gradients;

            // if (v_dot_g) < 0:
            //   V = (1-alpha) V - alpha |V|/|G| G
            //   |V| = length of V,
            //   if more than DELAYSTEP since v dot g was positive:
            //     increase timestep and decrease alpha
            if (v_dot_g < 0.)
              {
                const real_type velocities_norm_squared =
                  velocities * velocities;

                // Check if we devide by zero in DEBUG mode.
                Assert (gradient_norm_squared > 0., ExcInternalError());

                // beta = - alpha |V|/|G|
                const real_type beta = -alpha *
                                       std::sqrt (velocities_norm_squared
                                                  /
                                                  gradient_norm_squared);

                // V = (1-alpha) V + beta G.
                velocities.sadd (1. - alpha, beta, gradients);

                if (iter - previous_iter_with_positive_v_dot_g > DELAYSTEP)
                  {
                    timestep = std::min (timestep*TIMESTEP_GROW, max_timestep);
                    alpha *= ALPHA_SHRINK;
                  }
              }
            // else
            // decrease timestep, reset alpha and set V = 0
            else
              {
                previous_iter_with_positive_v_dot_g = iter;
                timestep *= TIMESTEP_SHRINK;
                alpha = ALPHA_0;
                velocities = 0.;
              }

            // Change timestep if any dof would move more than dmax?
            minimal_timestep = additional_data.dmax
                               /
                               velocities.linfty_norm();

            if (timestep < minimal_timestep)
              minimal_timestep = timestep;

            ++iter;

            print_vectors(iter, u, velocities, gradients);

          } // while didn't converge
      }
    catch (...)
      {
        this->memory.free(v);
        this->memory.free(g);
        throw;
      }

    this->memory.free(v);
    this->memory.free(g);
  }



  template <typename VectorType>
  void
  SolverFIRE<VectorType>::print_vectors(const unsigned int,
                                        const VectorType &,
                                        const VectorType &,
                                        const VectorType &) const
  {}

} // namespace statics


#endif // DOXYGEN


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_minimizer_fire_h */
