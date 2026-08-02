#ifndef PARAMSIM_CASES_CUBE_CASE_PARAMETERS_HPP_
#define PARAMSIM_CASES_CUBE_CASE_PARAMETERS_HPP_

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace paramsim {
namespace cases {
namespace cube {

/**
 * @brief Strict line-oriented reader for cube-case parameter files.
 *
 * Parsing errors identify the case, filename, and source line.
 */
class ParameterFileReader {
public:
    /**
     * @brief Opens a parameter file for parsing.
     *
     * @param case_name Name used to identify the case in error messages.
     * @param filename Path of the parameter file.
     * @throws std::runtime_error if the file cannot be opened.
     */
    ParameterFileReader(std::string case_name, std::string filename);

    /**
     * @brief Reads a strictly positive integer from the next line.
     *
     * @param description Human-readable name of the count.
     * @return The parsed count as an unsigned integer.
     * @throws std::runtime_error if the line is missing or malformed.
     */
    unsigned read_count(const std::string& description);

    /**
     * @brief Reads an exact number of finite values from the next line.
     *
     * @param expected_count Required number of values.
     * @param description Human-readable name of the row.
     * @return Parsed values in input order.
     * @throws std::runtime_error for missing, extra, or non-finite values.
     */
    std::vector<double> read_finite_values(std::size_t expected_count,
                                           const std::string& description);

    /**
     * @brief Throws a consistently formatted parsing error.
     *
     * @param line_number One-based source line associated with the error.
     * @param message Description of the parsing failure.
     */
    [[noreturn]] void fail(std::size_t line_number,
                           const std::string& message) const;

    /**
     * @brief Verifies that no non-whitespace input remains.
     *
     * @throws std::runtime_error if trailing data is present.
     */
    void require_end();

private:
    /**
     * @brief Reads the next required physical line.
     *
     * @param description Human-readable name of the expected line.
     * @return The line contents without the terminating newline.
     * @throws std::runtime_error if the line is missing or unreadable.
     */
    std::string read_required_line(const std::string& description);

    std::string case_name_;
    std::string filename_;
    std::ifstream input_;
    std::size_t line_number_{0};
};

/**
 * @brief Exponents for each active coordinate in a monomial.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
using MonomialExponent = std::array<unsigned int, dim>;

/**
 * @brief Enumerates all monomial exponents at one total degree.
 *
 * Tuples are ordered by descending x exponent and then descending y
 * exponent. In 3D, the z exponent is the remaining degree.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 * @param degree Requested total degree.
 * @return Exponents in deterministic coefficient order.
 */
template <int dim>
std::vector<MonomialExponent<dim>> degree_exponents(unsigned int degree);

/**
 * @brief Returns the coefficient count across several degree levels.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 * @param degree_levels Number of levels beginning with degree zero.
 * @return Total number of monomials in all requested levels.
 * @throws std::overflow_error if the total cannot fit in `std::size_t`.
 */
template <int dim>
std::size_t total_coefficient_count(unsigned int degree_levels);

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
