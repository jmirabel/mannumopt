#pragma once

#include <ostream>
#include <iomanip>

namespace mannumopt {

namespace internal {

template <typename ValueType>
inline void hprint(std::ostream& os, const char* header, const ValueType&)
{
  os << std::setfill(' ') << std::setw(12) << header << '\n';
}

template <typename ValueType>
inline void vprint(std::ostream& os, const char*, const ValueType& v)
{
  os << std::setprecision(5) << std::setw(12) << v << '\n';
}

template <typename ValueType, typename ...Args>
inline void hprint(std::ostream& os, const char* header, const ValueType&, const Args&... args)
{
  os << std::setfill(' ') << std::setw(12) << header;
  hprint(os, args...);
}

template <typename ValueType, typename ...Args>
inline void vprint(std::ostream& os, const char* header, const ValueType& v, const Args&... args)
{
  os << std::setfill(' ') << std::setw(12) << v;
  vprint(os, args...);
}

}

template <typename ...Args>
inline void print(std::ostream& os, bool print_headers, const Args&... args)
{
  if (print_headers)
    internal::hprint(os, args...);
  internal::vprint(os, args...);
}

}
