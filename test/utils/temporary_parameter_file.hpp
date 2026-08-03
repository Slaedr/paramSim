#ifndef PARAMSIM_TEST_UTILS_TEMPORARY_PARAMETER_FILE_HPP_
#define PARAMSIM_TEST_UTILS_TEMPORARY_PARAMETER_FILE_HPP_

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace paramsim {
namespace test {

/**
 * @brief Owns a uniquely named temporary parameter file used by tests.
 */
class TemporaryParameterFile {
public:
    /**
     * @brief Creates a temporary file containing supplied text.
     *
     * @param contents Text to write to the temporary file.
     * @throws std::runtime_error if the file cannot be created.
     */
    explicit TemporaryParameterFile(const std::string& contents)
    {
        static std::random_device random_source;
        path_ = std::filesystem::temp_directory_path() /
                ("paramsim_test_parameters_" +
                 std::to_string(random_source()) + "_" +
                 std::to_string(random_source()) + ".txt");
        std::ofstream output(path_);
        output << contents;
        if (!output) {
            throw std::runtime_error(
                "Could not create temporary parameter file.");
        }
    }

    /** @brief Removes the temporary parameter file without throwing. */
    ~TemporaryParameterFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    /**
     * @brief Returns the temporary parameter-file path.
     *
     * @return Path suitable for passing to a parameter reader.
     */
    std::string path() const
    {
        return path_.string();
    }

private:
    std::filesystem::path path_;
};

} // namespace test
} // namespace paramsim

#endif
