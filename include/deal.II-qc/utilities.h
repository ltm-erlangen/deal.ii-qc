
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

# include <boost/preprocessor/facilities/empty.hpp>
# include <boost/preprocessor/list/at.hpp>
# include <boost/preprocessor/list/for_each_product.hpp>
# include <boost/preprocessor/tuple/elem.hpp>
# include <boost/preprocessor/tuple/to_list.hpp>

#include <deal.II/base/numbers.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/dofs/dof_accessor.h>



// Preprocessor definitions for instantiations.
#define       DIM  BOOST_PP_TUPLE_TO_LIST(3,  (1,2,3))
#define  SPACEDIM  BOOST_PP_TUPLE_TO_LIST(3,  (1,2,3))
#define ATOMICITY  BOOST_PP_TUPLE_TO_LIST(10, (1,2,3,4,5,6,7,8,9,10))



// Accessors for SPACEDIM, ATOMICITY in (two element tuple) X.
#define  FIRST_OF_TWO_IS_SPACEDIM(X)    BOOST_PP_TUPLE_ELEM(2, 0, X)
#define SECOND_OF_TWO_IS_ATOMICITY(X)   BOOST_PP_TUPLE_ELEM(2, 1, X)



/**
 * A macro function that returns 1 if @p DIM is less or equal to @p SPACEDIM
 * and 0 otherwise among three entries DIM, ATOMICITY and, SPACEDIM.
 */
#define IS_DIM_LESS_EQUAL_SPACEDIM(DIM, ATOMICITY, SPACEDIM) \
  BOOST_PP_LESS_EQUAL(DIM, SPACEDIM)



/**
 * A macro function that returns 1 if @p DIM is less or equal to @p SPACEDIM
 * and 0 otherwise.
 */
#define IS_DIM_AND_SPACEDIM_PAIR_VALID(DIM, SPACEDIM) \
  BOOST_PP_LESS_EQUAL(DIM, SPACEDIM)



/**
 * A macro to expand CLASS macro using (DIM, SPACEDIM) tuples.
 */
#define INSTANTIATE_WITH_DIM_AND_SPACEDIM(CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 2, (DIM, SPACEDIM))



/**
 * A macro to expand CLASS macro using (SPACEDIM, ATOMICITY) tuples.
 */
#define INSTANTIATE_WITH_SPACEDIM_AND_ATOMICITY(R, CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 2, (SPACEDIM, ATOMICITY))



/**
 * A macro to expand CLASS macro using (DIM, SPACEDIM, ATOMICITY) tuples.
 */
#define INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 3, (DIM, ATOMICITY, SPACEDIM))



/**
 * Macro to suppress unused variable warnings.
 */
#define DEAL_II_QC_UNUSED_VARIABLE(X) ((void)(X))



/***************************************************************************
 * Two macro names that we put at the top and bottom of all deal.II-qc files
 * and that will be expanded to "namespace dealiiqc {" and "}".
 */
#define DEAL_II_QC_NAMESPACE_OPEN   namespace dealiiqc {
#define DEAL_II_QC_NAMESPACE_CLOSE  }


DEAL_II_QC_NAMESPACE_OPEN


/**
 * Custom types used inside dealiiqc.
 */
namespace types
{
  /**
   * The type used for global indices of atoms.
   * In order to have 64-bit unsigned integers (more than 4 billion),
   * build deal.II with support for 64-bit integers.
   * The data type always indicates an unsigned integer type.
   */
  typedef  dealii::types::global_dof_index global_atom_index;

  /**
   * The type used for global indices of molecules.
   */
  typedef  global_atom_index global_molecule_index;

  // TODO: Use of correct charge units; Use charge_t for book keeping.
  /**
   * The type used for storing charge of the atom.
   * Computations with charge of atoms don't need high precision.
   */
  typedef float charge;

  /**
   * The type used for identifying atom types. The enumeration starts
   * from 0.
   */
  typedef unsigned char atom_type;

  /**
   * A typedef for Triangulation's active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using CellIteratorType =
    typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;

  /**
   * A typedef for DoFHandler's active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using DoFCellIteratorType =
    typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;

  /**
   * A typedef for DoFHandler's const active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using ConstCellIteratorType =
    const CellIteratorType<dim, spacedim>;

} //typedefs



/**
 * A namespace for certain fixed numbers.
 */
namespace numbers
{
  /**
   * A number representing invalid cluster weight.
   */
  static const double
  invalid_cluster_weight = std::numeric_limits<double>::signaling_NaN();

} // numbers



/**
 * Make sure that sscanf doesn't pickup spaces as unsigned char
 * while parsing atom data stream.
 */
#define UC_SCANF_STR "%hhu"

namespace Utilities
{
  using namespace dealii;



  // TODO: Add a test for this function.
  /**
   * Find the closest vertex of a given cell @p cell to a given Point @p and
   * return a pair of its number and the squared distance.
   */
  template<int dim>
  inline
  std::pair<unsigned int, double>
  find_closest_vertex (const Point<dim> &p,
                       const typename Triangulation<dim>::cell_iterator cell)
  {
    // Throw exception if the given cell is is not in a valid
    // cell iterator state.
    AssertThrow (cell->state() == IteratorState::valid,
                 ExcMessage ("The given cell iterator is not in a "
                             "valid iterator state"));

    // Assume the first vertex is the closest at first.
    double squared_distance = p.distance_square(cell->vertex(0));
    unsigned int vertex_number = 0;

    // Loop over all the other vertices to search for closest vertex.
    for (unsigned int v=1; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const double p_squared_distance = p.distance_square(cell->vertex(v));
        if (p_squared_distance < squared_distance)
          {
            squared_distance = p_squared_distance;
            vertex_number = v;
          }
      }
    return std::make_pair(vertex_number, squared_distance);
  }



  /**
   * Find the closest point in a given list of @p points to a point @p p
   * and return a pair of its (the closest point's) index in the list and
   * the squared distance of separation.
   */
  template<int dim>
  inline
  std::pair<unsigned int, double>
  find_closest_point (const Point<dim>              &p,
                      const std::vector<Point<dim>> &points)
  {
    AssertThrow (points.size(),
                 ExcMessage("The given list of points is empty."));

    // Assume the first point is the closest at first.
    double squared_distance = points[0].distance_square(p);
    unsigned int point_index = 0;

    // Loop over all the other points to search for closest points.
    for (unsigned int i=1; i<points.size(); ++i)
      {
        const double p_squared_distance = points[i].distance_square(p);
        if (p_squared_distance < squared_distance)
          {
            squared_distance = p_squared_distance;
            point_index = i;
          }
      }

    return std::make_pair(point_index, squared_distance);
  }



  /**
   * Utility function that returns true if a point @p p is outside a bounding
   * box. The box is specified by two points @p minp and @p maxp (the order of
   * specifying points is important).
   */
  template<int dim>
  inline
  bool
  is_outside_bounding_box( const Point<dim> &minp,
                           const Point<dim> &maxp,
                           const Point<dim> &p)
  {
    for (unsigned int d=0; d<dim; ++d)
      if ( (minp[d] > p[d]) || (p[d] > maxp[d]) )
        {
          return true;
        }

    return false;
  }



  /**
   * Given a string that contains a list of @p minor_delimiter separated
   * text separated by a @p major_delimiter, split it into its components,
   * remove leading and trailing spaces.
   *
   * The input string can end without specifying any delimiters (possibly
   * followed by an arbitrary amount of whitespace). For example,
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l; ", ';', ',');
   * @endcode
   * yields the same 3-element list of sub-lists output
   * <code>{ {"abc", "def"},{"ghi"}, {"j", "k", "l"}}</code>
   * as you would get if the input had been
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l", ';', ',');
   * @endcode
   * or
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l;", ';', ',');
   * @endcode
   */
  inline
  std::vector<std::vector<std::string> >
  split_list_of_string_lists (const std::string &s,
                              const char major_delimiter = ';',
                              const char minor_delimiter = ',')
  {
    AssertThrow (major_delimiter!=minor_delimiter,
                 ExcMessage("Invalid major and minor delimiters provided!"));

    std::vector<std::vector<std::string>> res;

    const std::vector<std::string> coeffs_per_type =
      dealii::Utilities::split_string_list (s,
                                            major_delimiter);
    res.resize(coeffs_per_type.size());
    for (unsigned int i = 0; i < coeffs_per_type.size(); ++i)
      res[i] = dealii::Utilities::split_string_list (coeffs_per_type[i],
                                                     minor_delimiter);
    return res;
  }



  /**
   * Return the range of atom types on parsing a given string
   * @p numeric_string that consists a possible wildcard asterisk and the
   * total number of atom types @p n_atom_types. There are only five
   * possible ways to describe a range of atom types using @p numeric_string:
   * a) "i"   to describe the range [i, i+1)
   * b) "*"   to describe the range [0, @p max_atom_types)
   * c) "i*"  to describe the range [i, @p max_atom_types)
   * d) "*i"  to describe the range [0, i+1)
   * e) "i*j" to describe the range [i, j+1)
   */
  std::pair<types::atom_type, types::atom_type>
  atom_type_range (const std::string      &numeric_string,
                   const types::atom_type  n_atom_types);



  /**
   * Return the dim-dimensional volume of the hyper-ball segment, the region
   * of the hyper-ball being cutoff by a hyper-plane that is @p d distance
   * from the center of the hyper-ball of radius @p radius.
   *
   * @note The permissible range of @p d is [0, radius].
   * When the value of @p d is 0 the volume of the half hyper-ball
   * is returned.
   */
  template <int dim>
  inline
  double hyperball_segment_volume (const double &radius, const double &d)
  {
    double volume;

    // Height of the segment.
    const double height = radius-d;

    AssertThrow (0 <= height && height <= 2.*radius,
                 ExcMessage("This function is called with invalid parameter: "
                            "d, the distance from the center"
                            "of the hyper-ball to the hyper-plane."
                            "Allowed range of d is [0, radius]."));
    if (dim==1)
      volume = height;
    else if (dim==2)
      {
        // Half of the angle inscribed at the center of the hyper-ball.
        const double alpha = std::acos(d/radius);
        volume = radius* (radius*alpha - d*std::sin(alpha));
      }
    else if (dim==3)
      volume = dealii::numbers::PI * height * height * (3*radius - height)/3.;

    return volume;
  }

} // Utilities


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_utility_h */
