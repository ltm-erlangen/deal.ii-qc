// a source file which contains definition of core functions of QC class
#include <dealiiqc/qc.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  Atom<dim>::Atom ()
  {
    // initialize Points to clearly unusable values:
    for (unsigned int d = 0; d < dim; d++)
      {
        position[d] = (1./0.);
        reference_position[d] = (1./0.);
      }

    // TODO: init parent_cell
  }

  // instantiations:
  template class Atom<1>;
  template class Atom<2>;
  template class Atom<3>;
}
