
#ifndef __dealii_qc_rol_vector_adaptor_h
#define __dealii_qc_rol_vector_adaptor_h

#include "ROL_Vector.hpp"

#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace rol
{

  /**
   * Vector adaptor that provides @tparam VectorType implementation of the
   * ROL::Vector interface.
   *
   * VectorAdaptor supports any vector that satisfies @tparam VectorType concept
   * introduced in deal.II (see Vector classes in deal.II).
   */
  template<typename VectorType>
  class VectorAdaptor : public ROL::Vector<typename VectorType::real_type>
  {

    /**
     * A typedef for size type of VectorType.
     */
    using size_type = typename VectorType::size_type;

    /**
     * A typedef for size type of VectorType.
     */
    using real_type = typename VectorType::real_type;

  private:

    /**
     * Teuchos smart reference counting pointer to the Vector.
     */
    Teuchos::RCP<VectorType> vector_ptr;

  public:

    /**
     * Constructor.
     */
    VectorAdaptor (const Teuchos::RCP<VectorType> &vector_ptr);

    /**
     * Returns the Teuchos smart reference counting pointer to the Vector,
     * #vector_ptr.
     */
    Teuchos::RCP<VectorType> getVector ();

    /**
     * Returns the Teuchos smart reference counting pointer to const Vector.
     */
    Teuchos::RCP<const VectorType> getVector () const;

    /**
     * Return the dimension of the Vector.
     */
    int dimension () const;

    /**
     * Set the Vector to the given ROL::Vector @p rol_vector by copying its
     * contents to the Vector.
     */
    void set (const ROL::Vector<real_type> &rol_vector);

    /**
     * Perform addition.
     */
    void plus (const ROL::Vector<real_type> &rol_vector);

    /**
     * Scale the Vector by @p alpha and add ROL::Vector @p rol_vector to it.
     */
    void axpy (const real_type               alpha,
               const ROL::Vector<real_type> &rol_vector);

    /**
     * Scale the Vector.
     */
    void scale (const real_type alpha);

    /**
     * Return the dot product with a given ROL::Vector @p rol_vector.
     */
    real_type dot( const ROL::Vector<real_type> &rol_vector ) const;

    /**
     * Return the \f$ L_2 $\f norm of the Vector.
     */
    real_type norm() const;

    /**
     * Return a clone of the Vector.
     */
    Teuchos::RCP<ROL::Vector<real_type> > clone() const;

    /**
     * Return a Teuchos smart reference counting pointer the basis vector
     * corresponding to the @p i \f${}^{th}$\f element of the Vector.
     */
    Teuchos::RCP<ROL::Vector<real_type>> basis (const int i) const;

    /**
     * Apply unary function @p f to all the elements of the Vector.
     */
    void applyUnary (const ROL::Elementwise::UnaryFunction<real_type> &f);

    /**
     * Apply binary function @p f along with ROL::Vector @p x to all the
     * elements of the Vector.
     */
    void applyBinary (const ROL::Elementwise::UnaryFunction<real_type> &f,
                      const ROL::Vector<real_type>                     &x);

    /**
     * Return the accumulated value on applying reduction operation @p r on
     * all the elements of the Vector.
     */
    real_type reduce (const ROL::Elementwise::ReductionOp<real_type> &r) const;

    /**
     * Print the Vector to the output stream @p outStream.
     */
    void print (std::ostream &outStream) const;

  };

  /* --------------------- Inline and template functions ------------------- */
#ifndef DOXYGEN


  template<typename VectorType>
  VectorAdaptor<VectorType>::
  VectorAdaptor (const Teuchos::RCP<VectorType> &vector_ptr)
    :
    vector_ptr (vector_ptr)
  {}



  template<typename VectorType>
  Teuchos::RCP<VectorType>
  VectorAdaptor<VectorType>::getVector ()
  {
    return vector_ptr;
  }



  template<typename VectorType>
  Teuchos::RCP<const VectorType>
  VectorAdaptor<VectorType>::getVector () const
  {
    return vector_ptr;
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::set (const ROL::Vector<real_type> &rol_vector)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->size() != rol_vector.dimension(),
                                std::invalid_argument,
                                "Error: Vectors must have the same dimension.");

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    (*vector_ptr) =  *(vector_adaptor.getVector());
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::plus (const ROL::Vector<real_type> &rol_vector)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->size() != rol_vector.dimension(),
                                std::invalid_argument,
                                "Error: Vectors must have the same dimension.");

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    vector_ptr->add( *(vector_adaptor.getVector()) );
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::axpy (const real_type               alpha,
                                   const ROL::Vector<real_type> &rol_vector)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->size() != rol_vector.dimension(),
                                std::invalid_argument,
                                "Error: Vectors must have the same dimension." );

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    vector_ptr->add (alpha, *(vector_adaptor.getVector()));
  }



  template<typename VectorType>
  int
  VectorAdaptor<VectorType>::dimension () const
  {
    return vector_ptr->size();
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::scale (const real_type alpha)
  {
    (*vector_ptr) *= alpha;
  }



  template<typename VectorType>
  typename VectorType::real_type
  VectorAdaptor<VectorType>::
  dot (const ROL::Vector<real_type> &rol_vector) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->size() != rol_vector.dimension(),
                                std::invalid_argument,
                                "Error: Vectors must have the same dimension.");

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast< const VectorAdaptor>(rol_vector);

    return (*vector_ptr) * (*vector_adaptor.getVector());
  }



  template<typename VectorType>
  typename VectorType::real_type
  VectorAdaptor<VectorType>::norm() const
  {
    return vector_ptr->l2_norm();
  }



  template<typename VectorType>
  Teuchos::RCP<ROL::Vector<typename VectorType::real_type> >
  VectorAdaptor<VectorType>::clone() const
  {
    Teuchos::RCP< VectorType> vec_ptr = Teuchos::rcp (new VectorType);
    (*vec_ptr) = (*vector_ptr);

    return Teuchos::rcp (new VectorAdaptor (vec_ptr));
  }



  template<typename VectorType>
  Teuchos::RCP<ROL::Vector<typename VectorType::real_type> >
  VectorAdaptor<VectorType>::basis (const int i) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->locally_owned_elements().is_element(i),
                                std::invalid_argument,
                                "Error: Basis index must be between 0 and vector dimension.");
    Teuchos::RCP< VectorType> vec_ptr = Teuchos::rcp (new VectorType);

    // Zero all the entries in dealii vector.
    (*vec_ptr).reinit(*vector_ptr, false);

    Teuchos::RCP<VectorAdaptor> e =
      Teuchos::rcp( new VectorAdaptor( vec_ptr) );

    // Set asked basis.
    (*e->getVector())[i] = 1.0;

    return e;
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::
  applyUnary (const ROL::Elementwise::UnaryFunction<real_type> &f)
  {
    const dealii::IndexSet locally_owned_index_set =
      vector_ptr->locally_owned_elements ();

    for (unsigned int i = 0; i < locally_owned_index_set.n_elements(); ++i)
      {
        const unsigned int ind = locally_owned_index_set.nth_index_in_set(i);
        (*vector_ptr)[ind] = f.apply((*vector_ptr)[ind]);
      }
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::
  applyBinary (const ROL::Elementwise::UnaryFunction<real_type> &f,
               const ROL::Vector<real_type>                     &x)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (vector_ptr->size() != x.dimension(),
                                std::invalid_argument,
                                "Error: Vectors must have the same dimension.");

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(x);

    const VectorType &dealii_vector = *(vector_adaptor.getVector());

    const dealii::IndexSet locally_owned_index_set =
      vector_ptr->locally_owned_elements ();

    for (unsigned int i = 0; i < locally_owned_index_set.n_elements(); ++i)
      {
        const unsigned int ind = locally_owned_index_set.nth_index_in_set(i);
        (*vector_ptr)[ind] = f.apply((*vector_ptr)[ind], dealii_vector[ind]);
      }
  }



  template<typename VectorType>
  typename VectorType::real_type
  VectorAdaptor<VectorType>::
  reduce (const ROL::Elementwise::ReductionOp<real_type> &r) const
  {
    typename VectorType::real_type result = r.initialValue();

    const dealii::IndexSet locally_owned_index_set =
      vector_ptr->locally_owned_elements ();

    for (unsigned int i = 0; i < locally_owned_index_set.n_elements(); ++i)
      {
        const unsigned int ind = locally_owned_index_set.nth_index_in_set(i);
        r.reduce((*vector_ptr)[ind], result);
      }

    return result;
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::print (std::ostream &outStream) const
  {
    vector_ptr->print(outStream);
  }


#endif // DOXYGEN

} // namespace rol


DEAL_II_QC_NAMESPACE_CLOSE

#endif // __dealii_qc_rol_vector_adaptor_h
