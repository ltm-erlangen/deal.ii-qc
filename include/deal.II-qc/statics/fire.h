#ifndef __dealii_qc_minimizer_fire_h
#define __dealii_qc_minimizer_fire_h

#include <deal.II/lac/solver.h>

#include <deal.II-qc/utilities.h>

#include <functional>


DEAL_II_QC_NAMESPACE_OPEN


namespace statics
{

  // TODO: Complete the description.
  /**
   * A class implementation of FIRE (Fast Inertial Relaxation Engine) algorithm
   * which is a damped dynamics method described in <a href="">Structural
   * Relaxation Made Simple</a> by Bitzek et al. 2006 explained below.
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
  class SolverFIRE : public dealii::Solver<VectorType>
  {

  public:

    struct AdditionalData
    {
      explicit
      AdditionalData (const double  dt     = 1e-12,
                      const double  dt_max = 1e-9,
                      const double  dmax   = 1e-10,
                      std::shared_ptr<const typename dealii::DiagonalMatrix<VectorType>> inverse_masses = nullptr);

      /**
       * Time step.
       */
      double dt;

      /**
       * Maximum timestep.
       */
      const double dt_max;

      /**
       * Maximum change allowed in any degree of freedom.
       */
      const double dmax;

      /**
       * A const reference to the masses in the system.
       */
      const std::shared_ptr<const typename dealii::DiagonalMatrix<VectorType>> inverse_masses;

    };

    /**
     * Constructor.
     */
    SolverFIRE (dealii::SolverControl            &solver_control,
                dealii::VectorMemory<VectorType> &vector_memory,
                const AdditionalData             &data);

    SolverFIRE (dealii::SolverControl &solver_control,
                const AdditionalData  &data);

    /**
     * Virtual destructor.
     */
    virtual ~SolverFIRE();

    /**
     * Obtain a set of #p dofs (variables) that minimize an objective function
     * described by the polymorphic function wrapper @p compute. The function
     * @p compute takes in the state of the (dofs) variables as argument and
     * returns a pair of objective function's value and objective function's
     * gradient (with respect to the variables).
     */
    void solve
    (std::function<double(VectorType &, const VectorType &)>  compute,
     VectorType                                              &dofs);

  protected:

    /**
     * Additional parameters.
     */
    AdditionalData additional_data;

  };

  /* --------------------- Inline and template functions ------------------- */
#ifndef DOXYGEN

  template<typename VectorType>
  SolverFIRE<VectorType>::AdditionalData::
  AdditionalData (const double  dt,
                  const double  dt_max,
                  const double  dmax,
                  std::shared_ptr<const typename dealii::DiagonalMatrix<VectorType>> inverse_masses)
    :
    dt(dt),
    dt_max(dt_max),
    dmax(dmax),
    inverse_masses(inverse_masses)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::SolverFIRE (dealii::SolverControl            &solver_control,
                                      dealii::VectorMemory<VectorType> &vector_memory,
                                      const AdditionalData             &data)
    :
    dealii::Solver<VectorType>(solver_control, vector_memory),
    additional_data(data)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::SolverFIRE (dealii::SolverControl &solver_control,
                                      const AdditionalData  &data)
    :
    dealii::Solver<VectorType>(solver_control),
    additional_data(data)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::~SolverFIRE()
  {}



  template<typename VectorType>
  void
  SolverFIRE<VectorType>::solve
  (std::function<double(VectorType &, const VectorType &)>  compute,
   VectorType                                              &dofs)
  {
    // FIRE algorithm constants
    const double DELAYSTEP    = 5;
    const double DT_GROW      = 1.1;
    const double DT_SHRINK    = 0.5;
    const double ALPHA_0      = 0.1;
    const double ALPHA_SHRINK = 0.99;
    const double TMAX         = 10.0;

    using real_type = typename VectorType::real_type;
    using namespace dealii;

    VectorType *v, *g;

    v = this->memory.alloc();
    g = this->memory.alloc();

    VectorType &velocities = *v;
    VectorType &gradients  = *g;

    //FIXME: Initialize velocities adopting a normal distribution?
    velocities.reinit(dofs, false); // V = 0
    gradients.reinit(dofs, true);

    compute(dofs, gradients);

    unsigned int iter = 0;

    SolverControl::State conv = SolverControl::iterate;
    real_type gradient_norm_squared = gradients * gradients;

    // Check whether tolerance criteria is satisfied already.
    conv = this->iteration_status(0, gradient_norm_squared, dofs);
    if (conv != SolverControl::iterate)
      return;

    try
      {
        // First scaling factor.
        double alpha = ALPHA_0;

        // Scaling factors.
        double
        a = std::numeric_limits<double>::signaling_NaN(),
        b = std::numeric_limits<double>::signaling_NaN();

        unsigned int previous_iter_with_positive_v_dot_g = 0;

        while (conv == SolverControl::iterate)
          {

            // v_dot_g = V * G
            real_type v_dot_g = velocities * gradients;

            // if (v_dot_g) < 0:
            //   V = (1-alpha) V - alpha |V| Ghat
            //   |V| = length of V, Ghat = unit G
            //   if more than DELAYSTEP since v dot g was positive:
            //     increase timestep and decrease alpha
            if (v_dot_g < 0.)
              {
                real_type velocities_norm_squared =
                  velocities * velocities;

                a = 1. - alpha;
                if (gradient_norm_squared == 0.)
                  b = 0.;
                else
                  b = alpha * std::sqrt (velocities_norm_squared
                                         /
                                         gradient_norm_squared);

                velocities.sadd (a, b, gradients);

                if (iter - previous_iter_with_positive_v_dot_g > DELAYSTEP)
                  {
                    additional_data.dt = std::min (additional_data.dt*DT_GROW, additional_data.dt_max);
                    alpha *= ALPHA_SHRINK;
                  }
              }
            // else
            // decrease timestep, reset alpha and set V = 0
            else
              {
                previous_iter_with_positive_v_dot_g = iter;
                additional_data.dt *= DT_SHRINK;
                alpha = ALPHA_0;
                velocities = 0.;
              }

            // Check whether energy tolerance criteria is satisfied?
            //

            // Change timestep if any dof would be changed more than dmax.
            double minimal_timestep =
              additional_data.dmax
              /
              dealii::Utilities::MPI::max (velocities.max(),
                                           dofs.get_mpi_communicator());

            if (additional_data.dt < minimal_timestep)
              minimal_timestep = additional_data.dt;

            // Euler integration step.
            dofs.sadd (minimal_timestep, velocities);
            additional_data.inverse_masses->vmult(gradients, gradients);
            velocities.sadd(-minimal_timestep, gradients);

            // Update gradients for the new dofs.
            compute(dofs, gradients);

            ++iter;
            //print_vectors(iter, dofs, gradients);

            conv = this->iteration_status(iter, gradient_norm_squared, dofs);
            if (conv != SolverControl::iterate)
              break;

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

} // namespace statics


#endif // DOXYGEN


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_minimizer_fire_h */
