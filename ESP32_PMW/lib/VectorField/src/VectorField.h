#pragma once

#include <math.h>

namespace coilArray {
constexpr double ringRadiusMeters = 0.037;
constexpr double coilPitchMeters = 0.021;
constexpr double coilRadiusMeters = 0.0105;
constexpr double turnsPerCoil = 650.0;
constexpr double channelAzimuthDegrees[4] = {0.0, 90.0, 180.0, 270.0};
constexpr double phaseReferencePoint[3] = {0.0, 0.0, 0.015};
constexpr double seriesResistanceOhms = 6.9;
constexpr double seriesInductanceHenries = 1.4e-3;
constexpr double seriesCapacitanceFarads = 800e-6;
constexpr double railVolts = 12.0;
constexpr double continuousCurrentAmps = 2.0;
}

struct CoilSettings {
  float dutyPercent[4];
  float phaseDegrees[4];
  float fieldFraction;
};

class VectorField {
 public:
  explicit VectorField(const float spinPhaseDegrees[4]);

  CoilSettings settingsFor(const double point[3], const double orientation[3][3], double fieldTesla,
                           double frequencyHz) const;
  CoilSettings settingsFor(const double point[3], const double quaternion[4], double fieldTesla,
                           double frequencyHz) const;

  static bool rotationFromQuaternion(const double quaternion[4], double rotation[3][3]);
  void fieldPerAmp(const double point[3], double field[3][4]) const;
  bool currentsFor(const double point[3], const double orientation[3][3], double fieldTesla,
                   double realAmps[4], double imaginaryAmps[4]) const;
  CoilSettings settingsFromCurrents(const double realAmps[4], const double imaginaryAmps[4],
                                    double frequencyHz) const;

  double channelGain[4] = {1.0, 1.0, 1.0, 1.0};
  double phaseOffsetDegrees[4];

 private:
  double coilPositions[16][3];
};
