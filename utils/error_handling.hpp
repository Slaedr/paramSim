#ifndef PARAMSIM_ERROR_HANDLING_HPP_
#define PARAMSIM_ERROR_HANDLING_HPP_

#include <stdexcept>
#include <string>

namespace paramsim {

/// An error thrown when an unsupported dynamic type is encountered.
class TypeNotSupportedError : public std::runtime_error
{
public:
    TypeNotSupportedError(const std::string& msg)
        : std::runtime_error(std::string("Type not supported: ") + msg)
    { }
};

}

#endif // PARAMSIM_ERROR_HANDLING_HPP_
