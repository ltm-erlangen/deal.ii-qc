
#ifndef __dealii_qc_rol_vector_adaptor_h
#define __dealii_qc_rol_vector_adaptor_h

// FIXME: Require Trilinos and that it is configured with ROL.
#ifdef DEAL_II_WITH_TRILINOS
#include "ROL_Vector.hpp"

#include <deal.II-qc/utilities.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <type_traits>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;


/**
 * A namespace that provides an interface to the
 * <a href="https://trilinos.org/docs/dev/packages/rol/doc/html/index.html">
 * Rapid Optimization Library</a> (ROL).
 */
namespace rol
{

  /**
   * Vector adaptor that provides <tt>VectorType</tt> implementation of the
   * ROL::Vector interface.
   *
   * VectorAdaptor supports vectors that satisfies the following requirements.
   *
   * The VectorType should contain the following types.
   * @code
   * VectorType::size_type;  // The type for size of the vector.
   * VectorType::value_type; // The type for elements stored in the vector.
   * VectorType::real_type;  // The type for real-valued numbers.
   * @endcode
   *
   * However, ROL doesn't distinguish VectorAdaptor::value_type from
   * VectorAdaptor::real_type. This is due to ROL's assumption that the
   * VectorAdaptor::value_type itself is a type for real-valued numbers.
   * Therefore, VectorAdaptor supports vectors with whose real_type is
   * convertible to value_type in the sense that
   * `std::is_convertible<real_type, value_type>::value` yields `true`.
   *
   * The VectorType should contain the following methods.
   * @code
   *                   // Reinitialize the current vector using a given vector's
   *                   // size (and the parallel distribution) without copying
   *                   // the elements.
   * VectorType::reinit(const VectorType &, ...);
   *
   *                       // Globally add a given vector to the current.
   * VectorType::operator+=(const VectorType &);
   *
   *                       // Scale all elements by a given scalar.
   * VectorType::operator*=(const VectorType::value_type &);
   *
   *                       // Perform dot product with a given vector.
   * VectorType::operator*=(const VectorType &);
   *
   *                // Scale all elements of the current vector and globally
   *                // add a given vector to it.
   * VectorType::add(const VectorType::value_type, const VectorType &);
   *
   *                      // Copies the data of a given vector to the current.
   *                      // Resize the current vector if necessary (MPI safe).
   * VectorType::operation=(const VectorType &);
   *
   *                 // To query the global size of the current vector.
   * VectorType::size();
   *
   *                    // To query L_2 norm of the current vector
   * VectorType::l2_norm();
   *
   *                  // Iterator to the start of the (locally owned) element
   *                  // of the current vector.
   * VectorType::begin();
   *
   *                // Iterator to the one past the last (locally owned)
   *                // element of the current vector.
   * VectorType::end();
   *
   *                     // Compress the vector i.e., flush the buffers of the
   *                     // vector object if it has any.
   * VectorType::compress(VectorOperation::insert);
   * @endcode
   *
   * Most of the vectors in deal.II (see Vector classes) adhere to the above
   * requirements.
   *
   * The current implementation in ROL doesn't support vector sizes above
   * the largest value of int type.
   *
   *
   */
  template<typename VectorType>
  class VectorAdaptor : public ROL::Vector<typename VectorType::value_type>
  {

    /**
     * A typedef for size type of VectorType.
     */
    using size_type = typename VectorType::size_type;

    /**
     * A typedef for element type stored in the VectorType.
     */
    using value_type = typename VectorType::value_type;

    /**
     * A typedef for real-valued numbers.
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
    void set (const ROL::Vector<value_type> &rol_vector);

    /**
     * Perform addition.
     */
    void plus (const ROL::Vector<value_type> &rol_vector);

    /**
     * Scale the Vector by @p alpha and add ROL::Vector @p rol_vector to it.
     */
    void axpy (const value_type               alpha,
               const ROL::Vector<value_type> &rol_vector);

    /**
     * Scale the Vector.
     */
    void scale (const value_type alpha);

    /**
     * Return the dot product with a given ROL::Vector @p rol_vector.
     */
    value_type dot( const ROL::Vector<value_type> &rol_vector ) const;

    /**
     * Return the \f$ L_2 \f$ norm of the Vector.
     *
     * The returned type is of VectorAdaptor::value_type so as to maintain
     * consistency with ROL::Vector<VectorAdaptor::value_type> and
     * more importantly to not to create an overloaded version namely,
     * @code
     *   VectorAdaptor::real_type norm() const;
     * @endcode
     * if real_type and value_type are not of the same type .
     * with
     * VectorAdaptor::real_type return.
     */
    value_type norm() const;

    /**
     * Return a clone of the Vector.
     */
    Teuchos::RCP<ROL::Vector<value_type>> clone() const;

    /**
     * Create and return a Teuchos smart reference counting pointer to the basis
     * vector corresponding to the @p i \f${}^{th}\f$ element of the Vector.
     */
    Teuchos::RCP<ROL::Vector<value_type>> basis (const int i) const;

    /**
     * Apply unary function @p f to all the elements of the Vector.
     */
    void applyUnary (const ROL::Elementwise::UnaryFunction<value_type> &f);

    /**
     * Apply binary function @p f along with ROL::Vector @p rol_vector to all
     * the elements of the Vector.
     */
    void applyBinary (const ROL::Elementwise::UnaryFunction<value_type> &f,
                      const ROL::Vector<value_type>                     &rol_vector);

    /**
     * Return the accumulated value on applying reduction operation @p r on
     * all the elements of the Vector.
     */
    value_type reduce (const ROL::Elementwise::ReductionOp<value_type> &r) const;

    /**
     * Print the Vector to the output stream @p outStream.
     */
    void print (std::ostream &outStream) const;

  };


  /*------------------------------member definitions--------------------------*/
#ifndef DOXYGEN


  template<typename VectorType>
  VectorAdaptor<VectorType>::
  VectorAdaptor (const Teuchos::RCP<VectorType> &vector_ptr)
    :
    vector_ptr (vector_ptr)
  {
    Assert ((std::is_convertible<real_type, value_type>::value),
            ExcMessage("The value_type and the real_type of the current "
                       "VectorType being used aren't convertible."));
  }



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
  VectorAdaptor<VectorType>::set (const ROL::Vector<value_type> &rol_vector)
  {
    Assert (this->dimension() == rol_vector.dimension(),
            ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    (*vector_ptr) =  *(vector_adaptor.getVector());
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::plus (const ROL::Vector<value_type> &rol_vector)
  {
    Assert (this->dimension() == rol_vector.dimension(),
            ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    *vector_ptr += *(vector_adaptor.getVector());
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::axpy (const value_type               alpha,
                                   const ROL::Vector<value_type> &rol_vector)
  {
    Assert (this->dimension() == rol_vector.dimension(),
            ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    vector_ptr->add (alpha, *(vector_adaptor.getVector()));
  }



  template<typename VectorType>
  int
  VectorAdaptor<VectorType>::dimension () const
  {
    Assert (vector_ptr->size() < std::numeric_limits<int>::max(),
            ExcMessage("The size of the vector being used is greater than "
                       "largest value of type int."));
    return static_cast<int>(vector_ptr->size());
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::scale (const value_type alpha)
  {
    (*vector_ptr) *= alpha;
  }



  template<typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::
  dot (const ROL::Vector<value_type> &rol_vector) const
  {
    Assert (this->dimension() == rol_vector.dimension(),
            ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast< const VectorAdaptor>(rol_vector);

    return (*vector_ptr) * (*vector_adaptor.getVector());
  }



  template<typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::norm() const
  {
    return static_cast<value_type>(vector_ptr->l2_norm());
  }



  template<typename VectorType>
  Teuchos::RCP<ROL::Vector<typename VectorType::value_type> >
  VectorAdaptor<VectorType>::clone() const
  {
    Teuchos::RCP< VectorType> vec_ptr = Teuchos::rcp (new VectorType);
    (*vec_ptr) = (*vector_ptr);

    return Teuchos::rcp (new VectorAdaptor (vec_ptr));
  }



  template<typename VectorType>
  Teuchos::RCP<ROL::Vector<typename VectorType::value_type> >
  VectorAdaptor<VectorType>::basis (const int i) const
  {
    Teuchos::RCP<VectorType> vec_ptr = Teuchos::rcp (new VectorType);

    // Zero all the entries in dealii vector.
    vec_ptr->reinit(*vector_ptr, false);

    if (vector_ptr->locally_owned_elements().is_element(i))
      vec_ptr->operator[](i) = 1.;

    vec_ptr->compress(VectorOperation::insert);

    Teuchos::RCP<VectorAdaptor> e = Teuchos::rcp (new VectorAdaptor(vec_ptr));

    return e;
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::
  applyUnary (const ROL::Elementwise::UnaryFunction<value_type> &f)
  {
    for (typename VectorType::iterator
         iterator  = vector_ptr->begin();
         iterator != vector_ptr->end();
         iterator++)
      *iterator = f.apply(*iterator);

    vector_ptr->compress (VectorOperation::insert);
  }



  template<typename VectorType>
  void
  VectorAdaptor<VectorType>::
  applyBinary (const ROL::Elementwise::UnaryFunction<value_type> &f,
               const ROL::Vector<value_type>                     &rol_vector)
  {
    Assert (this->dimension() == rol_vector.dimension(),
            ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      Teuchos::dyn_cast<const VectorAdaptor>(rol_vector);

    const VectorType &dealii_vector = *(vector_adaptor.getVector());

    for (typename VectorType::iterator
         l_iterator  = vector_ptr->begin(), r_iterator  = dealii_vector.begin();
         l_iterator != vector_ptr->end() && r_iterator != dealii_vector.end();
         l_iterator++,                      r_iterator++)
      *l_iterator = f.apply(*l_iterator, *r_iterator);

    vector_ptr->compress (VectorOperation::insert);
  }



  template<typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::
  reduce (const ROL::Elementwise::ReductionOp<value_type> &r) const
  {
    typename VectorType::value_type result = r.initialValue();

    for (typename VectorType::iterator
         iterator  = vector_ptr->begin();
         iterator != vector_ptr->end();
         iterator++)
      r.reduce(*iterator, result);

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


#endif // DEAL_II_WITH_TRILINOS

#endif // __dealii_qc_rol_vector_adaptor_h
