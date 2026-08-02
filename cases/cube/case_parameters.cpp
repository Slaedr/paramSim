#include "case_parameters.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace paramsim {
namespace cases {
namespace cube {

ParameterFileReader::ParameterFileReader(std::string case_name,
                                         std::string filename)
    : case_name_{std::move(case_name)}, filename_{std::move(filename)},
      input_{filename_}
{
    if (!input_.is_open()) {
        throw std::runtime_error("Case '" + case_name_ +
                                 "': cannot read parameter file '" + filename_ +
                                 "'.");
    }
}

unsigned
ParameterFileReader::read_count(const std::string& description)
{
    const std::string line = read_required_line(description);
    std::istringstream values(line);
    long long count = 0;
    if (!(values >> count)) {
        fail(line_number_, "expected " + description +
                               " to be a positive integer");
    }

    values >> std::ws;
    if (!values.eof()) {
        fail(line_number_, "unexpected data after " + description);
    }
    if (count < 1) {
        fail(line_number_, description + " must be at least one");
    }

    const auto unsigned_count = static_cast<unsigned long long>(count);
    if (unsigned_count > std::numeric_limits<unsigned>::max()) {
        fail(line_number_, description + " is too large");
    }
    return static_cast<unsigned>(unsigned_count);
}

std::vector<double> ParameterFileReader::read_finite_values(
    const std::size_t expected_count, const std::string& description)
{
    const std::string line = read_required_line(description);
    std::istringstream values(line);
    std::vector<double> result;
    result.reserve(expected_count);

    for (std::size_t index = 0; index < expected_count; ++index) {
        double value = 0.0;
        if (!(values >> value)) {
            fail(line_number_, "expected " + std::to_string(expected_count) +
                                   " values for " + description);
        }
        if (!std::isfinite(value)) {
            fail(line_number_, description + " contains a non-finite value");
        }
        result.push_back(value);
    }

    values >> std::ws;
    if (!values.eof()) {
        fail(line_number_, "unexpected data after " + description);
    }
    return result;
}

void ParameterFileReader::require_end()
{
    std::string line;
    while (std::getline(input_, line)) {
        ++line_number_;
        std::istringstream trailing(line);
        trailing >> std::ws;
        if (!trailing.eof()) {
            fail(line_number_, "unexpected trailing data");
        }
    }
    if (input_.bad()) {
        fail(line_number_ + 1, "error while reading parameter file");
    }
}

std::string
ParameterFileReader::read_required_line(const std::string& description)
{
    std::string line;
    if (!std::getline(input_, line)) {
        if (input_.bad()) {
            fail(line_number_ + 1, "error while reading " + description);
        }
        fail(line_number_ + 1, "missing " + description);
    }
    ++line_number_;
    return line;
}

void ParameterFileReader::fail(const std::size_t line_number,
                               const std::string& message) const
{
    throw std::runtime_error("Case '" + case_name_ + "', parameter file '" +
                             filename_ + "', line " +
                             std::to_string(line_number) + ": " + message +
                             ".");
}

template <int dim>
std::vector<MonomialExponent<dim>>
degree_exponents(const unsigned int degree)
{
    static_assert(dim == 2 || dim == 3,
                  "Polynomial exponents require dimension 2 or 3.");
    std::vector<MonomialExponent<dim>> exponents;

    for (unsigned int x_exponent = degree;; --x_exponent) {
        const unsigned int remaining = degree - x_exponent;
        if constexpr (dim == 2) {
            exponents.push_back(
                MonomialExponent<dim>{{x_exponent, remaining}});
        } else {
            for (unsigned int y_exponent = remaining;; --y_exponent) {
                exponents.push_back(MonomialExponent<dim>{
                    {x_exponent, y_exponent, remaining - y_exponent}});
                if (y_exponent == 0) {
                    break;
                }
            }
        }
        if (x_exponent == 0) {
            break;
        }
    }
    return exponents;
}

template <int dim>
static
std::size_t degree_coefficient_count(const unsigned int degree)
{
    static_assert(dim == 2 || dim == 3,
                  "Polynomial coefficient counts require dimension 2 or 3.");
    return degree_exponents<dim>(degree).size();
}

template <int dim>
std::size_t total_coefficient_count(const unsigned int degree_levels)
{
    static_assert(dim == 2 || dim == 3,
                  "Polynomial coefficient counts require dimension 2 or 3.");
    std::size_t count = 0;
    for (unsigned int degree = 0; degree < degree_levels; ++degree) {
        const std::size_t degree_count =
            degree_coefficient_count<dim>(degree);
        if (degree_count > std::numeric_limits<std::size_t>::max() - count) {
            throw std::overflow_error("Polynomial coefficient count overflow.");
        }
        count += degree_count;
    }
    return count;
}

template std::vector<MonomialExponent<2>>
degree_exponents<2>(unsigned int);
template std::vector<MonomialExponent<3>>
degree_exponents<3>(unsigned int);

template std::size_t total_coefficient_count<2>(unsigned int);
template std::size_t total_coefficient_count<3>(unsigned int);

} // namespace cube
} // namespace cases
} // namespace paramsim
