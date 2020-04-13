
#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Utilities
{
  using namespace dealii;


  std::pair<types::atom_type, types::atom_type>
  atom_type_range(const std::string &    numeric_string,
                  const types::atom_type n_atom_types)
  {
    const std::string numeric_string_trimmed =
      dealii::Utilities::trim(numeric_string);

    const std::size_t string_size = numeric_string_trimmed.size();

    AssertThrow(string_size > 0, ExcEmptyObject());

    types::atom_type minimum = 0;
    types::atom_type maximum = n_atom_types - 1;

    const bool found_asterisk =
      (numeric_string_trimmed.find('*') != std::string::npos);

    // Handle simple number case and simple asterisk case directly.
    if (!found_asterisk)
      {
        minimum = dealii::Utilities::string_to_int(numeric_string_trimmed);
        return {minimum, minimum};
      }
    else if (found_asterisk && string_size == 1)
      // If the input string is single asterisk return the full range.
      return {minimum, maximum};

    const std::vector<std::string> s =
      dealii::Utilities::split_string_list(numeric_string_trimmed, '*');

    Assert(s.size() <= 2,
           ExcMessage("Invalid atom type range specified: " + numeric_string));

    if (s.size() == 1)
      minimum = dealii::Utilities::string_to_int(s[0]);
    else if (s.size() == 2)
      {
        if (!s[0].empty())
          minimum = dealii::Utilities::string_to_int(s[0]);

        if (!s[1].empty())
          maximum = dealii::Utilities::string_to_int(s[1]);
      }

    Assert(minimum < n_atom_types && maximum < n_atom_types,
           ExcMessage("Invalid atom type range specified: " + numeric_string));

    return {minimum, maximum};
  }

} // namespace Utilities


DEAL_II_QC_NAMESPACE_CLOSE
