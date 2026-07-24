#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "../../cases/cube/case_parameters.hpp"

namespace cube = paramsim::cases::cube;

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
        path_ = std::filesystem::temp_directory_path() /
                ("paramsim_case_parameters_" + std::to_string(next_id++) +
                 ".txt");
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
    EXPECT_EQ(cube::degree_exponents<2>(0),
              std::vector<E>({E{{0, 0}}}));
    EXPECT_EQ(cube::degree_exponents<2>(1),
              std::vector<E>({E{{1, 0}}, E{{0, 1}}}));
    EXPECT_EQ(cube::degree_exponents<2>(2),
              std::vector<E>({E{{2, 0}}, E{{1, 1}}, E{{0, 2}}}));
    EXPECT_EQ(cube::degree_exponents<2>(3),
              std::vector<E>(
                  {E{{3, 0}}, E{{2, 1}}, E{{1, 2}}, E{{0, 3}}}));
}

TEST(PolynomialExponents, EnumeratesThreeDimensionsThroughDegreeThree)
{
    using E = cube::MonomialExponent<3>;
    EXPECT_EQ(cube::degree_exponents<3>(0),
              std::vector<E>({E{{0, 0, 0}}}));
    EXPECT_EQ(cube::degree_exponents<3>(1),
              std::vector<E>(
                  {E{{1, 0, 0}}, E{{0, 1, 0}}, E{{0, 0, 1}}}));
    EXPECT_EQ(cube::degree_exponents<3>(2),
              std::vector<E>({E{{2, 0, 0}},
                              E{{1, 1, 0}},
                              E{{1, 0, 1}},
                              E{{0, 2, 0}},
                              E{{0, 1, 1}},
                              E{{0, 0, 2}}}));
    EXPECT_EQ(cube::degree_exponents<3>(3),
              std::vector<E>({E{{3, 0, 0}},
                              E{{2, 1, 0}},
                              E{{2, 0, 1}},
                              E{{1, 2, 0}},
                              E{{1, 1, 1}},
                              E{{1, 0, 2}},
                              E{{0, 3, 0}},
                              E{{0, 2, 1}},
                              E{{0, 1, 2}},
                              E{{0, 0, 3}}}));
}

TEST(PolynomialExponents, ReportsDegreeAndTotalCoefficientCounts)
{
    EXPECT_EQ(cube::degree_coefficient_count<2>(3), 4);
    EXPECT_EQ(cube::degree_coefficient_count<3>(3), 10);
    EXPECT_EQ(cube::total_coefficient_count<2>(4), 10);
    EXPECT_EQ(cube::total_coefficient_count<3>(4), 20);
}

TEST(ParameterFileReader, ReadsStrictCountsAndFiniteRows)
{
    TemporaryParameterFile file("2\n1.25 -3.5\n0 4\n");
    cube::ParameterFileReader reader("cube_test", file.path());

    EXPECT_EQ(reader.read_count("number of rows"), 2);
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

TEST(ParameterFileReader, RejectsUnreadableFiles)
{
    EXPECT_THROW(
        cube::ParameterFileReader("cube_test",
                                  "/path/that/does/not/exist/params.txt"),
        std::runtime_error);
}
