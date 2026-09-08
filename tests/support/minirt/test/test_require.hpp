#pragma once

#include <exception>
#include <source_location>
#include <string>
#include <utility>

namespace minirt::test {

class Failure final : public std::exception {
  public:
    explicit Failure(std::string message) : message_(std::move(message)) {}

    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

  private:
    std::string message_;
};

inline void Require(bool condition, const char* expression,
                    std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw Failure(std::string(location.file_name()) + ':' + std::to_string(location.line()) +
                      ": requirement failed: " + expression);
    }
}

} // namespace minirt::test

// Host-side check: evaluates the expression once and remains active in Release builds.
#define MINIRT_TEST_REQUIRE(...)                                                                   \
    ::minirt::test::Require(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__)
