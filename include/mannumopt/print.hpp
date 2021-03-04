#pragma once

#include <ostream>
#include <iomanip>

namespace mannumopt {

namespace internal {

inline void hprint(std::ostream& os) { os << '\n'; }

inline void vprint(std::ostream& os) { os << '\n'; }

template <typename ValueType, typename ...Args>
void hprint(std::ostream& os, const char* header, const ValueType&, const Args&... args);

template <typename ValueType, typename ...Args>
void vprint(std::ostream& os, const char* header, const ValueType& v, const Args&... args);

template <typename ValueType, typename ...Args>
inline void hprint(std::ostream& os, int w, const char* header, const ValueType&, const Args&... args)
{
  os << std::setfill(' ') << std::setw(w) << header;
  hprint(os, args...);
}

template <typename ValueType, typename ...Args>
inline void vprint(std::ostream& os, int w, const char* header, const ValueType& v, const Args&... args)
{
  os << std::setfill(' ') << std::setw(w) << v;
  vprint(os, args...);
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
