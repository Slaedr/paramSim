#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "../../cases/cube/case_parameters.hpp"
#include "../../cases/cube/exponential.hpp"
#include "../../cases/cube/fourier.hpp"
#include "../../cases/cube/polynomial.hpp"

namespace cube = paramsim::cases::cube;
namespace exponential = cube::exponential;
namespace fourier = cube::fourier;
namespace polynomial = cube::polynomial;

namespace {

/**
 * @brief Owns a temporary parameter file used by one test.
 */
class TemporaryParameterFile {
public:
    /**
     * @brief Creates a uniquely named temporary file with supplied contents.
     *
     * @param contents Text to write to the temporary file.
     * @throws std::runtime_error if the file cannot be created.
     */
    explicit TemporaryParameterFile(const std::string& contents)
    {
        static unsigned int next_id = 0;
        path_ =
            std::filesystem::temp_directory_path() /
            ("paramsim_case_parameters_" + std::to_string(next_id++) + ".txt");
        std::ofstream output(path_);
        output << contents;
        if (!output) {
            throw std::runtime_error("Could not create temporary test file.");
        }
    }

    /**
     * @brief Removes the temporary file without throwing.
     */
    ~TemporaryParameterFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    /**
     * @brief Returns the temporary file path as a string.
     *
     * @return Path suitable for passing to the parameter reader.
     */
    std::string path() const
    {
        return path_.string();
    }

private:
    std::filesystem::path path_;
};

} // namespace

TEST(PolynomialExponents, EnumeratesTwoDimensionsThroughDegreeThree)
{
    using E = cube::MonomialExponent<2>;
    EXPECT_EQ(cube::degree_exponents<2>(0), std::vector<E>({E{{0, 0}}}));
    EXPECT_EQ(cube::degree_exponents<2>(1),
              std::vector<E>({E{{1, 0}}, E{{0, 1}}}));
    EXPECT_EQ(cube::degree_exponents<2>(2),
              std::vector<E>({E{{2, 0}}, E{{1, 1}}, E{{0, 2}}}));
    EXPECT_EQ(cube::degree_exponents<2>(3),
              std::vector<E>({E{{3, 0}}, E{{2, 1}}, E{{1, 2}}, E{{0, 3}}}));
}

TEST(PolynomialExponents, EnumeratesThreeDimensionsThroughDegreeThree)
{
    using E = cube::MonomialExponent<3>;
    EXPECT_EQ(cube::degree_exponents<3>(0), std::vector<E>({E{{0, 0, 0}}}));
    EXPECT_EQ(cube::degree_exponents<3>(1),
              std::vector<E>({E{{1, 0, 0}}, E{{0, 1, 0}}, E{{0, 0, 1}}}));
    EXPECT_EQ(cube::degree_exponents<3>(2),
              std::vector<E>({E{{2, 0, 0}}, E{{1, 1, 0}}, E{{1, 0, 1}},
                              E{{0, 2, 0}}, E{{0, 1, 1}}, E{{0, 0, 2}}}));
    EXPECT_EQ(
        cube::degree_exponents<3>(3),
        std::vector<E>({E{{3, 0, 0}}, E{{2, 1, 0}}, E{{2, 0, 1}}, E{{1, 2, 0}},
                        E{{1, 1, 1}}, E{{1, 0, 2}}, E{{0, 3, 0}}, E{{0, 2, 1}},
                        E{{0, 1, 2}}, E{{0, 0, 3}}}));
}

TEST(PolynomialExponents, ReportsDegreeAndTotalCoefficientCounts)
{
    EXPECT_EQ(cube::degree_exponents<2>(3).size(), 4);
    EXPECT_EQ(cube::degree_exponents<3>(3).size(), 10);
    EXPECT_EQ(cube::total_coefficient_count<2>(4), 10);
    EXPECT_EQ(cube::total_coefficient_count<3>(4), 20);
}

TEST(ParameterFileReader, ReadsStrictCountsAndFiniteRows)
{
    TemporaryParameterFile file("2\n1.25 -3.5\n0 4\n");
    cube::ParameterFileReader reader("cube_test", file.path());

    auto row_count = reader.read_count("number of rows");
    static_assert(std::is_same_v<decltype(row_count), unsigned>);
    EXPECT_EQ(row_count, 2U);
    EXPECT_EQ(reader.read_finite_values(2, "first row"),
              std::vector<double>({1.25, -3.5}));
    EXPECT_EQ(reader.read_finite_values(2, "second row"),
              std::vector<double>({0.0, 4.0}));
    EXPECT_NO_THROW(reader.require_end());
}

TEST(ParameterFileReader, RejectsInvalidCounts)
{
    for (const std::string contents : {"0\n", "-2\n", "1.5\n", "2 extra\n"}) {
        TemporaryParameterFile file(contents);
        cube::ParameterFileReader reader("cube_test", file.path());
        EXPECT_THROW(reader.read_count("number of rows"), std::runtime_error);
    }
}

TEST(ParameterFileReader, RejectsIncorrectAndNonFiniteRows)
{
    for (const std::string contents :
         {"1\n1\n", "1\n1 2 3\n", "1\nnan 2\n", "1\ninf 2\n"}) {
        TemporaryParameterFile file(contents);
        cube::ParameterFileReader reader("cube_test", file.path());
        ASSERT_EQ(reader.read_count("number of rows"), 1);
        EXPECT_THROW(reader.read_finite_values(2, "data row"),
                     std::runtime_error);
    }
}

TEST(ParameterFileReader, RejectsMissingRowsAndTrailingData)
{
    {
        TemporaryParameterFile file("1\n");
        cube::ParameterFileReader reader("cube_test", file.path());
        ASSERT_EQ(reader.read_count("number of rows"), 1);
        EXPECT_THROW(reader.read_finite_values(2, "data row"),
                     std::runtime_error);
    }
    {
        TemporaryParameterFile file("1\n1 2\ntrailing\n");
        cube::ParameterFileReader reader("cube_test", file.path());
        ASSERT_EQ(reader.read_count("number of rows"), 1);
        reader.read_finite_values(2, "data row");
        EXPECT_THROW(reader.require_end(), std::runtime_error);
    }
}

TEST(ParameterFileReader, ErrorsIdentifyCaseFileAndLine)
{
    TemporaryParameterFile file("1\n1\n");
    cube::ParameterFileReader reader("cube_named_case", file.path());
    ASSERT_EQ(reader.read_count("number of rows"), 1);

    try {
        reader.read_finite_values(2, "data row");
        FAIL() << "Expected malformed row to throw.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("cube_named_case"), std::string::npos);
        EXPECT_NE(message.find(file.path()), std::string::npos);
        EXPECT_NE(message.find("line 2"), std::string::npos);
    }
}

TEST(ExponentialParameters, ParsesRuntimeSizedCentersInBothDimensions)
{
    TemporaryParameterFile file("2\n"
                                "-1 -0.5 0.75 2 0.25\n"
                                "0.5 1 -0.75 -3 0.8\n");

    const auto params_2d = exponential::read_parameters<2>(file.path());
    const auto params_3d = exponential::read_parameters<3>(file.path());

    ASSERT_EQ(params_2d.centers.size(), 2);
    ASSERT_EQ(params_3d.centers.size(), 2);
    EXPECT_EQ(params_2d.centers[0].coordinates,
              (std::array<double, 2>{{-1.0, -0.5}}));
    EXPECT_EQ(params_3d.centers[1].coordinates,
              (std::array<double, 3>{{0.5, 1.0, -0.75}}));
    EXPECT_DOUBLE_EQ(params_2d.centers[0].coefficient, 2.0);
    EXPECT_DOUBLE_EQ(params_2d.centers[0].width, 0.25);
    EXPECT_DOUBLE_EQ(params_3d.centers[1].coefficient, -3.0);
    EXPECT_DOUBLE_EQ(params_3d.centers[1].width, 0.8);
}

TEST(ExponentialParameters, UsesIndependentWidthsAndIgnoresZ)
{
    TemporaryParameterFile file("2\n"
                                "0 0 -1 2 0.5\n"
                                "0 0 1 -1 1\n");
    const exponential::DirichletIn<2> profile_2d(
        exponential::read_parameters<2>(file.path()));
    const exponential::DirichletIn<3> profile_3d(
        exponential::read_parameters<3>(file.path()));

    const double expected = 3.5 / dealii::numbers::PI;
    EXPECT_NEAR(profile_2d.value(dealii::Point<2>{0.0, 0.0}), expected, 1e-14);
    EXPECT_NEAR(profile_3d.value(dealii::Point<3>{0.0, 0.0, 0.25}), expected,
                1e-14);
}

TEST(ExponentialParameters, PreservesBuiltInDefaults)
{
    const exponential::Params<2> params_2d;
    const exponential::Params<3> params_3d;

    ASSERT_EQ(params_2d.centers.size(), 3);
    ASSERT_EQ(params_3d.centers.size(), 3);
    EXPECT_EQ(params_2d.centers[0].coordinates,
              (std::array<double, 2>{{-1.0, -0.67}}));
    EXPECT_EQ(params_3d.centers[2].coordinates,
              (std::array<double, 3>{{-1.0, 0.66, 0.0}}));
    EXPECT_DOUBLE_EQ(params_2d.centers[0].coefficient, 0.27);
    EXPECT_DOUBLE_EQ(params_2d.centers[0].width, 0.4);
    EXPECT_DOUBLE_EQ(params_3d.centers[2].coefficient, -0.34);
    EXPECT_DOUBLE_EQ(params_3d.centers[2].width, 0.4);
}

TEST(ExponentialParameters, RejectsMalformedRowsAndTrailingData)
{
    for (const std::string contents :
         {"0\n", "1\n0 0 0 1\n", "1\n0 0 0 1 0.5 extra\n",
          "1\n0 0 0 1 0.5\ntrailing\n"}) {
        TemporaryParameterFile file(contents);
        EXPECT_THROW(exponential::read_parameters<2>(file.path()),
                     std::runtime_error);
    }
}

TEST(ExponentialParameters, RejectsInvalidCoordinatesAndWidths)
{
    for (const std::string contents :
         {"1\n-1.01 0 0 1 0.5\n", "1\n0 1.01 0 1 0.5\n", "1\n0 0 -1.01 1 0.5\n",
          "1\n0 0 0 1 0\n", "1\n0 0 0 1 -0.5\n"}) {
        TemporaryParameterFile file(contents);
        EXPECT_THROW(exponential::read_parameters<2>(file.path()),
                     std::runtime_error);
    }
}

TEST(FourierParameters, ParsesRuntimeSizedModesInBothDimensions)
{
    TemporaryParameterFile file("3\n"
                                "1.5 0.75\n"
                                "1 -2\n"
                                "3 -4\n"
                                "5 -6\n");

    const auto params_2d = fourier::read_parameters<2>(file.path());
    const auto params_3d = fourier::read_parameters<3>(file.path());

    ASSERT_EQ(params_2d.modes.size(), 3);
    ASSERT_EQ(params_3d.modes.size(), 3);
    EXPECT_DOUBLE_EQ(params_2d.constant, 1.5);
    EXPECT_DOUBLE_EQ(params_3d.fundamental_wavelength, 0.75);
    EXPECT_DOUBLE_EQ(params_2d.modes[0].cosine_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(params_2d.modes[0].sine_coefficient, -2.0);
    EXPECT_DOUBLE_EQ(params_3d.modes[2].cosine_coefficient, 5.0);
    EXPECT_DOUBLE_EQ(params_3d.modes[2].sine_coefficient, -6.0);
}

TEST(FourierParameters, AssignsRowsToFrequenciesStartingAtOne)
{
    TemporaryParameterFile file("2\n"
                                "0 2\n"
                                "1 0\n"
                                "1 0\n");
    const fourier::DirichletIn<2> profile_2d(
        fourier::read_parameters<2>(file.path()));
    const fourier::DirichletIn<3> profile_3d(
        fourier::read_parameters<3>(file.path()));

    const double expected = std::sqrt(0.5);
    EXPECT_NEAR(profile_2d.value(dealii::Point<2>{0.0, 0.25}), expected, 1e-14);
    EXPECT_NEAR(profile_3d.value(dealii::Point<3>{0.0, 0.25, 0.8}), expected,
                1e-14);
}

TEST(FourierParameters, PreservesBuiltInDefaults)
{
    const fourier::Params<2> params_2d;
    const fourier::Params<3> params_3d;

    ASSERT_EQ(params_2d.modes.size(), 2);
    ASSERT_EQ(params_3d.modes.size(), 2);
    EXPECT_DOUBLE_EQ(params_2d.constant, 1.0);
    EXPECT_DOUBLE_EQ(params_3d.fundamental_wavelength, 1.0);
    EXPECT_DOUBLE_EQ(params_2d.modes[0].cosine_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(params_2d.modes[0].sine_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(params_3d.modes[1].cosine_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(params_3d.modes[1].sine_coefficient, 1.0);
}

TEST(FourierParameters, RejectsMalformedRowsAndTrailingData)
{
    for (const std::string contents :
         {"0\n", "1\n1\n1 2\n", "1\n1 0.5 extra\n1 2\n", "2\n1 0.5\n1 2\n",
          "1\n1 0.5\n1 2\ntrailing\n"}) {
        TemporaryParameterFile file(contents);
        EXPECT_THROW(fourier::read_parameters<2>(file.path()),
                     std::runtime_error);
    }
}

TEST(FourierParameters, RejectsNonpositiveWavelengths)
{
    for (const std::string contents : {"1\n1 0\n1 2\n", "1\n1 -0.5\n1 2\n"}) {
        TemporaryParameterFile file(contents);
        EXPECT_THROW(fourier::read_parameters<3>(file.path()),
                     std::runtime_error);
    }
}

TEST(PolynomialParameters, ParsesDimensionSpecificRows)
{
    TemporaryParameterFile file_2d("3\n"
                                   "0.5 -1.5 99\n"
                                   "1\n"
                                   "2 3\n"
                                   "4 5 6\n");
    TemporaryParameterFile file_3d("4\n"
                                   "-1 2 -3\n"
                                   "1\n"
                                   "2 3 4\n"
                                   "5 6 7 8 9 10\n"
                                   "11 12 13 14 15 16 17 18 19 20\n");

    const auto params_2d = polynomial::read_parameters<2>(file_2d.path());
    const auto params_3d = polynomial::read_parameters<3>(file_3d.path());

    EXPECT_EQ(params_2d.center, (std::array<double, 2>{{0.5, -1.5}}));
    EXPECT_EQ(params_3d.center, (std::array<double, 3>{{-1.0, 2.0, -3.0}}));
    EXPECT_EQ(
        params_2d.coefficients_by_degree,
        (std::vector<std::vector<double>>{{1.0}, {2.0, 3.0}, {4.0, 5.0, 6.0}}));
    EXPECT_EQ(
        params_3d.coefficients_by_degree,
        (std::vector<std::vector<double>>{
            {1.0},
            {2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}));
}

TEST(PolynomialParameters, EvaluatesMixedTermsAboutNonzeroCenter)
{
    TemporaryParameterFile file("4\n"
                                "1 2 3\n"
                                "0\n"
                                "0 0 0\n"
                                "0 2 3 0 5 0\n"
                                "0 0 0 0 7 0 0 0 0 0\n");

    const polynomial::DirichletIn<3> profile(
        polynomial::read_parameters<3>(file.path()));

    EXPECT_DOUBLE_EQ(profile.value(dealii::Point<3>{3.0, 5.0, 7.0}), 264.0);
}

TEST(PolynomialParameters, TwoDimensionsIgnoresThirdCenterCoordinate)
{
    TemporaryParameterFile first_file("3\n"
                                      "1 2 99\n"
                                      "0\n"
                                      "0 0\n"
                                      "0 2 0\n");
    TemporaryParameterFile second_file("3\n"
                                       "1 2 -99\n"
                                       "0\n"
                                       "0 0\n"
                                       "0 2 0\n");

    const auto first_params = polynomial::read_parameters<2>(first_file.path());
    const auto second_params =
        polynomial::read_parameters<2>(second_file.path());
    const polynomial::DirichletIn<2> first_profile(first_params);
    const polynomial::DirichletIn<2> second_profile(second_params);

    EXPECT_EQ(first_params.center, second_params.center);
    EXPECT_EQ(first_params.coefficients_by_degree,
              second_params.coefficients_by_degree);
    EXPECT_DOUBLE_EQ(first_profile.value(dealii::Point<2>{3.0, 5.0}), 12.0);
    EXPECT_DOUBLE_EQ(second_profile.value(dealii::Point<2>{3.0, 5.0}), 12.0);
}

TEST(PolynomialParameters, PreservesBuiltInDefaults)
{
    const polynomial::Params<2> params_2d;
    const polynomial::Params<3> params_3d;
    const polynomial::DirichletIn<2> profile_2d(params_2d);
    const polynomial::DirichletIn<3> profile_3d(params_3d);

    EXPECT_EQ(params_2d.center, (std::array<double, 2>{}));
    EXPECT_EQ(params_3d.center, (std::array<double, 3>{}));
    EXPECT_EQ(params_2d.coefficients_by_degree,
              (std::vector<std::vector<double>>{
                  {1.0}, {0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0}}));
    EXPECT_EQ(params_3d.coefficients_by_degree,
              (std::vector<std::vector<double>>{
                  {1.0},
                  {0.0, 1.0, 0.0},
                  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}}));
    EXPECT_DOUBLE_EQ(profile_2d.value(dealii::Point<2>{-0.4, 2.0}), 15.0);
    EXPECT_DOUBLE_EQ(profile_3d.value(dealii::Point<3>{-0.4, 2.0, 0.7}), 15.0);
}

TEST(PolynomialParameters, RejectsMalformedFiles)
{
    for (const std::string contents :
         {"0\n", "1\n0 0\n1\n", "1\n0 0 0 extra\n1\n", "1\n0 0 0\n",
          "1\nnan 0 0\n1\n", "1\n0 0 0\nnan\n", "1\n0 0 0\n1 extra\n",
          "1\n0 0 0\n1\ntrailing\n"}) {
        TemporaryParameterFile file(contents);
        EXPECT_THROW(polynomial::read_parameters<2>(file.path()),
                     std::runtime_error);
    }
}

TEST(PolynomialParameters, RejectsDimensionallyIncorrectCoefficientCounts)
{
    TemporaryParameterFile too_many_for_2d("2\n"
                                           "0 0 0\n"
                                           "1\n"
                                           "2 3 4\n");
    TemporaryParameterFile too_few_for_3d("2\n"
                                          "0 0 0\n"
                                          "1\n"
                                          "2 3\n");

    EXPECT_THROW(polynomial::read_parameters<2>(too_many_for_2d.path()),
                 std::runtime_error);
    EXPECT_THROW(polynomial::read_parameters<3>(too_few_for_3d.path()),
                 std::runtime_error);
}
