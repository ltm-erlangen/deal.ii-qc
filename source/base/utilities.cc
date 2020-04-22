
#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Utilities
{
  using namespace dealii;


  std::pair<types::atom_type, types::atom_type>
  atom_type_range(const std::string &    numeric_string,
                  const types::atom_type n_atom_types)
  {
    std::string trimmed_numeric_string{numeric_string};

    // Remove all white spaces from the string.
    trimmed_numeric_string.erase(std::remove(trimmed_numeric_string.begin(),
                                             trimmed_numeric_string.end(),
                                             ' '),
                                 trimmed_numeric_string.end());

    AssertThrow(trimmed_numeric_string.size() > 0, ExcEmptyObject());

    if (trimmed_numeric_string.find('*') == std::string::npos)
      // asterisk not found
      return {dealii::Utilities::string_to_int(trimmed_numeric_string) - 1,
              dealii::Utilities::string_to_int(trimmed_numeric_string) - 1};

    const std::vector<std::string> s =
      dealii::Utilities::split_string_list(trimmed_numeric_string, '*');

    Assert(s.size() <= 2,
           ExcMessage("Invalid atom type range specified: " +
                      trimmed_numeric_string));

    const types::atom_type minimum =
      s[0].empty() ? 0 : dealii::Utilities::string_to_int(s[0]) - 1;

    types::atom_type maximum = n_atom_types - 1;

    if (s.size() == 2 && s[1].size())
      maximum = dealii::Utilities::string_to_int(s[1]) - 1;

    // Check that both minimum and maximum are within range.
    Assert(minimum <= maximum && maximum <= n_atom_types - 1,
           ExcMessage("Invalid atom type range specified: " +
                      trimmed_numeric_string));

    return {minimum, maximum};
  }

} // namespace Utilities


DEAL_II_QC_NAMESPACE_CLOSE
