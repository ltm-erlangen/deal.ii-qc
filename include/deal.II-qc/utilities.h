
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

namespace dealiiqc
{
  /**
   * Custom types used inside dealiiqc.
   */
  namespace types
  {
    /**
     * The type used for global indices of atoms.
     * In order to have 64-bit unsigned integers (more than 4 billion),
     *  build deal.II with support for 64-bit integers.
     *  The data type always indicates an unsigned integer type.
     */
    typedef  dealii::types::global_dof_index global_atom_index;

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

  } //typedefs

  /**
   * Make sure that sscanf doesn't pickup spaces as unsigned char
   * while parsing atom data stream.
   */
#define UC_SCANF_STR "%hhu"

  namespace Utilities
  {
    using namespace dealii;



    /**
     * Function to check if a Point @p p is a within a certain
     * @p distance from the vertices of a given parent cell.
     */
    template<int dim>
    inline
    bool
    is_point_within_distance_from_cell_vertices( const Point<dim> &p,
                                                 const typename Triangulation<dim>::cell_iterator cell,
                                                 const double &distance)
    {
      // Throw exception if the given cell is is not in a valid
      // cell iterator state.
      AssertThrow( cell->state() == IteratorState::valid,
                   ExcMessage( "The given cell iterator is not in a valid iterator state"));

      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        if (  p.distance_square(cell->vertex(v))
              < dealii::Utilities::fixed_power<2>( distance ) )
          return true;
      return false;
    }



    /**
     * Utility function that returns true if a point @p p is outside a bounding box.
     * The box is specified by two points @p minp and @p maxp (the order of
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
     *   Utilities::split_list_of_string_lists("abc, def; ghi; j, k, l; ", ';', ',');
     * @endcode
     * yields the same 3-element list of sub-lists output
     * <code>{ {"abc", "def"},{"ghi"}, {"j", "k", "l"}}</code>
     * as you would get if the input had been
     * @code
     *   Utilities::split_list_of_string_lists("abc, def; ghi; j, k, l", ';', ',');
     * @endcode
     * or
     * @code
     *   Utilities::split_list_of_string_lists("abc, def; ghi; j, k, l;", ';', ',');
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



  } // Utilities

}

#endif /* __dealii_qc_utility_h */
