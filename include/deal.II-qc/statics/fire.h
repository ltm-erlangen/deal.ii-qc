#ifndef __dealii_qc_minimizer_fire_h
#define __dealii_qc_minimizer_fire_h

#include <deal.II/lac/solver.h>

#include <deal.II-qc/statics/objective.h>

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
      AdditionalData (const double               dt,
                      const double               dt_max,
                      const std::vector<double> &masses);

      /**
       * Time step.
       */
      const double dt;

      /**
       * Maximum timestep.
       */
      const double dt_max;

      /**
       * A const reference to the masses in the system.
       */
      const std::vector<double> masses;

    };

    /**
     * Constructor.
     */
    SolverFIRE (dealii::SolverControl            &solver_control,
                dealii::VectorMemory<VectorType> &vector_memory,
                const AdditionalData             &data);

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
    (std::function<std::pair<double, VectorType>(const VectorType &)> compute,
     VectorType                                                       &dofs  );

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
  AdditionalData (const double               dt,
                  const double               dt_max,
                  const std::vector<double> &masses)
    :
    dt(dt),
    dt_max(dt_max),
    masses(masses)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::SolverFIRE (dealii::SolverControl            &solver_control,
                                      dealii::VectorMemory<VectorType> &vector_memory,
                                      const AdditionalData               &data)
    :
    Solver<VectorType>(solver_control, vector_memory),
    additional_data(data)
  {}



  template<typename VectorType>
  SolverFIRE<VectorType>::~SolverFIRE()
  {}



  template<typename VectorType>
  void
  SolverFIRE<VectorType>::solve
  (std::function< std::pair<double, VectorType>(const VectorType &)>  compute,
   VectorType                                                        &dofs    )
  {
    // FIRE algorithm constants
    const double DELAYSTEP    = 5;
    const double DT_GROW      = 1.1;
    const double DT_SHRINK    = 0.5;
    const double ALPHA_0      = 0.1;
    const double ALPHA_SHRINK = 0.99;
    const double TMAX         = 10.0;

    // TODO: Complete the algorithm.
  }

} // namespace statics


#endif // DOXYGEN


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_minimizer_fire_h */
