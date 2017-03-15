// a source file which contains definition of core functions of QC class
#include <dealiiqc/atom/atom.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  Atom<dim>::Atom ()
  {}

  // instantiations:
  template class Atom<1>;
  template class Atom<2>;
  template class Atom<3>;
}
