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
  template<typename VectorType, typename PreconditionerType = DiagonalMatrix<VectorType>>
  class SolverFIRE : public Solver<VectorType>
  {

  public:

    struct AdditionalData
    {
      //FIXME: Introduce units for timestep and umax.
      explicit
      AdditionalData (const double  given_timestep = 1e-12,
                      const double  maximum_timestep   = 1e-9,
                      const double  umax               = 1e-10);

      /**
       * Suggested time step.
       */
      const double given_timestep;

      /**
       * Maximum time step.
       */
      const double maximum_timestep;

      /**
       * Maximum change allowed in any degree of freedom.
       */
      const double umax;

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
     * described by the polymorphic function wrapper @p compute, with a given
     * preconditioner @p inverse_masses and initial @p u values.
     * The function @p compute takes in the state of the (u) variables as
     * argument and returns a pair of objective function's value and
     * objective function's gradient (with respect to the variables).
     */
    void solve
    (std::function<double(VectorType &, const VectorType &)>  compute,
     VectorType                                              &u,
     const PreconditionerType                                &inverse_masses);

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

  template<typename VectorType, typename PreconditionerType>
  SolverFIRE<VectorType, PreconditionerType>::AdditionalData::
  AdditionalData (const double  given_timestep,
                  const double  maximum_timestep,
                  const double  umax)
    :
    given_timestep(given_timestep),
    maximum_timestep(maximum_timestep),
    umax(umax)
  {}



  template<typename VectorType, typename PreconditionerType>
  SolverFIRE<VectorType, PreconditionerType>::
  SolverFIRE (SolverControl            &solver_control,
              VectorMemory<VectorType> &vector_memory,
              const AdditionalData     &data          )
    :
    Solver<VectorType>(solver_control, vector_memory),
    additional_data(data)
  {}



  template<typename VectorType, typename PreconditionerType>
  SolverFIRE<VectorType, PreconditionerType>::
  SolverFIRE (SolverControl         &solver_control,
              const AdditionalData  &data          )
    :
    Solver<VectorType>(solver_control),
    additional_data(data)
  {}



  template<typename VectorType, typename PreconditionerType>
  SolverFIRE<VectorType, PreconditionerType>::~SolverFIRE()
  {}



  template<typename VectorType, typename PreconditionerType>
  void
  SolverFIRE<VectorType, PreconditionerType>::solve
  (std::function<double(VectorType &, const VectorType &)>  compute,
   VectorType                                              &u,
   const PreconditionerType                                &inverse_masses)
  {
    // FIRE algorithm constants
    const double DELAYSTEP       = 5;
    const double TIMESTEP_GROW   = 1.1;
    const double TIMESTEP_SHRINK = 0.5;
    const double ALPHA_0         = 0.1;
    const double ALPHA_SHRINK    = 0.99;

    using real_type = typename VectorType::real_type;

    typename VectorMemory<VectorType>::Pointer v(this->memory);
    typename VectorMemory<VectorType>::Pointer g(this->memory);

    // Set velocities to zero but not gradients
    // as we are going to compute them soon.
    v->reinit(u,false);
    g->reinit(u,true);

    // Refer to v and g with some readable names.
    VectorType &velocities = *v;
    VectorType &gradients  = *g;

    // Update gradients for the new u.
    compute(gradients, u);

    unsigned int iter = 0;

    SolverControl::State conv = SolverControl::iterate;
    conv = this->iteration_status (iter, gradients * gradients, u);
    if (conv != SolverControl::iterate)
      return;

    // Refer to additional data members with some readable names.
    const auto &maximum_timestep   = additional_data.maximum_timestep;
    double timestep                = additional_data.given_timestep;

    // First scaling factor.
    double alpha = ALPHA_0;

    unsigned int previous_iter_with_positive_v_dot_g = 0;

    while (conv == SolverControl::iterate)
      {
        // Euler integration step.
        u.sadd (timestep, velocities);               // U += dt     * V
        inverse_masses.vmult(gradients, gradients);  // G  = M^{-1} * G
        velocities.sadd(-timestep, gradients);       // V -= dt     * G

        // Compute gradients for the new u.
        compute(gradients, u);

        const real_type gradient_norm_squared = gradients * gradients;
        conv = this->iteration_status(iter, gradient_norm_squared, u);
        if (conv != SolverControl::iterate)
          break;

        // v_dot_g = V * G
        const real_type v_dot_g = velocities * gradients;

        if (v_dot_g < 0.)
          {
            const real_type velocities_norm_squared =
              velocities * velocities;

            // Check if we divide by zero in DEBUG mode.
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
                // Increase timestep and decrease alpha.
                timestep = std::min (timestep*TIMESTEP_GROW, maximum_timestep);
                alpha *= ALPHA_SHRINK;
              }
          }
        else
          {
            // Decrease timestep, reset alpha and set V = 0.
            previous_iter_with_positive_v_dot_g = iter;
            timestep *= TIMESTEP_SHRINK;
            alpha = ALPHA_0;
            velocities = 0.;
          }

        real_type vmax = velocities.linfty_norm();

        // Change timestep if any dof would move more than umax?
        if (vmax > 0.)
          {
            const double minimal_timestep = additional_data.umax
                                            /
                                            vmax;
            if (minimal_timestep < timestep)
              timestep = minimal_timestep;
          }

        ++iter;

        print_vectors(iter, u, velocities, gradients);

      } // While we need to iterate.

    // In the case of failure: throw exception.
    if (conv != SolverControl::success)
      AssertThrow (false,
                   SolverControl::NoConvergence (iter, gradients * gradients));

  }



  template <typename VectorType, typename PreconditionerType>
  void
  SolverFIRE<VectorType, PreconditionerType>::
  print_vectors (const unsigned int,
                 const VectorType &,
                 const VectorType &,
                 const VectorType &) const
  {}

} // namespace statics


#endif // DOXYGEN


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_minimizer_fire_h */
