// Copyright (c) 2026 Jetacker contributors
#pragma once
#include <array>
// Captured entry 001848/index2071, reduced into the body frame.
namespace entry_steering_fixture
{
inline constexpr std::array<float, 2> speed = {{
  -7.9999999518e-02f, -1.4884312636e-01f
}};
inline constexpr std::array<float, 2> steering = {{
  4.8826211691e-01f, 4.8827317357e-01f
}};
inline constexpr std::array<float, 4> history_vx = {{
  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
  2.0966051146e-02f
}};
inline constexpr std::array<float, 4> history_wz = {{
  0.0000000000e+00f, 0.0000000000e+00f, 0.0000000000e+00f,
  -4.8688985407e-02f
}};
inline constexpr std::array<float, 8> path_x = {{
  -9.3027781476e-03f, 3.8528183610e-02f, 8.5846730406e-02f,
  1.3229288706e-01f, 1.8160228604e-01f, 2.3091168503e-01f,
  2.8022101201e-01f, 3.2953060100e-01f
}};
inline constexpr std::array<float, 8> path_y = {{
  6.7123183109e-03f, 2.9196870027e-03f, -5.0272907599e-03f,
  -1.7068340178e-02f, -2.5349698076e-02f, -3.3631055975e-02f,
  -4.1912508879e-02f, -5.0194010787e-02f
}};
inline constexpr std::array<float, 8> path_yaw = {{
  -3.5494692213e-02f, -1.2276119698e-01f, -2.1002764214e-01f,
  -2.9729408730e-01f, -1.6639489640e-01f, -1.6639489640e-01f,
  -1.6639489640e-01f, -1.6639489640e-01f
}};
}  // namespace entry_steering_fixture
