#include "VectorField.h"

using namespace coilArray;

static const double magneticConstantOverFourPi = 1e-7;
static const double dipoleMomentPerAmp = turnsPerCoil * M_PI * coilRadiusMeters * coilRadiusMeters;
static const double degreesPerRadian = 180.0 / M_PI;

static double wrapDegrees(double degrees) {
  degrees = fmod(degrees, 360.0);
  return degrees < 0 ? degrees + 360.0 : degrees;
}

static double dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static void cross(const double a[3], const double b[3], double result[3]) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

static bool isRotation(const double matrix[3][3]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (!(fabs(dot(matrix[i], matrix[j]) - (i == j)) < 1e-3)) return false;
  double lastTwoRowsCrossed[3];
  cross(matrix[1], matrix[2], lastTwoRowsCrossed);
  return fabs(dot(matrix[0], lastTwoRowsCrossed) - 1.0) < 1e-3;
}

VectorField::VectorField(const float spinPhaseDegrees[4]) {
  int coil = 0;
  for (int channel = 0; channel < 4; channel++) {
    const double azimuth = channelAzimuthDegrees[channel] / degreesPerRadian;
    const double radial[2] = {cos(azimuth), sin(azimuth)}, tangential[2] = {-radial[1], radial[0]};
    for (int radialSide = -1; radialSide <= 1; radialSide += 2)
      for (int tangentialSide = -1; tangentialSide <= 1; tangentialSide += 2, coil++) {
        for (int axis = 0; axis < 2; axis++)
          coilPositions[coil][axis] = ringRadiusMeters * radial[axis] +
                                      0.5 * coilPitchMeters * (radialSide * radial[axis] + tangentialSide * tangential[axis]);
        coilPositions[coil][2] = 0.0;
      }
  }

  const double identity[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double realAmps[4], imaginaryAmps[4];
  currentsFor(phaseReferencePoint, identity, 1.0, realAmps, imaginaryAmps);
  for (int channel = 0; channel < 4; channel++)
    phaseOffsetDegrees[channel] =
        wrapDegrees(spinPhaseDegrees[channel] + atan2(imaginaryAmps[channel], realAmps[channel]) * degreesPerRadian);
}

CoilSettings VectorField::settingsFor(const double point[3], const double orientation[3][3], double fieldTesla,
                                      double frequencyHz) const {
  double realAmps[4], imaginaryAmps[4];
  if (!currentsFor(point, orientation, fieldTesla, realAmps, imaginaryAmps)) return CoilSettings{};
  return settingsFromCurrents(realAmps, imaginaryAmps, frequencyHz);
}

CoilSettings VectorField::settingsFor(const double point[3], const double quaternion[4], double fieldTesla,
                                      double frequencyHz) const {
  double orientation[3][3];
  if (!rotationFromQuaternion(quaternion, orientation)) return CoilSettings{};
  return settingsFor(point, orientation, fieldTesla, frequencyHz);
}

bool VectorField::rotationFromQuaternion(const double quaternion[4], double rotation[3][3]) {
  const double length = sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                             quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  if (!(length > 0.0)) return false;
  const double w = quaternion[0] / length, x = quaternion[1] / length;
  const double y = quaternion[2] / length, z = quaternion[3] / length;
  rotation[0][0] = 1 - 2 * (y * y + z * z); rotation[0][1] = 2 * (x * y - w * z);     rotation[0][2] = 2 * (x * z + w * y);
  rotation[1][0] = 2 * (x * y + w * z);     rotation[1][1] = 1 - 2 * (x * x + z * z); rotation[1][2] = 2 * (y * z - w * x);
  rotation[2][0] = 2 * (x * z - w * y);     rotation[2][1] = 2 * (y * z + w * x);     rotation[2][2] = 1 - 2 * (x * x + y * y);
  return true;
}

void VectorField::fieldPerAmp(const double point[3], double field[3][4]) const {
  for (int axis = 0; axis < 3; axis++)
    for (int channel = 0; channel < 4; channel++) field[axis][channel] = 0.0;
  for (int coil = 0; coil < 16; coil++) {
    const double offset[3] = {point[0] - coilPositions[coil][0], point[1] - coilPositions[coil][1],
                              point[2] - coilPositions[coil][2]};
    const double distance = sqrt(dot(offset, offset));
    const double momentAlongOffset = offset[2] * dipoleMomentPerAmp;
    for (int axis = 0; axis < 3; axis++)
      field[axis][coil / 4] += magneticConstantOverFourPi * 3.0 * momentAlongOffset * offset[axis] / pow(distance, 5);
    field[2][coil / 4] -= magneticConstantOverFourPi * dipoleMomentPerAmp / pow(distance, 3);
  }
}

bool VectorField::currentsFor(const double point[3], const double orientation[3][3], double fieldTesla,
                              double realAmps[4], double imaginaryAmps[4]) const {
  // b is signed: -b gives I -> -I, a 180 deg phase shift on every channel (a twist by pi).
  // NaN/inf still refused because vec= arrives over serial.
  if (!isfinite(fieldTesla) || !isRotation(orientation)) return false;

  double field[3][4], gram[3][3], adjugate[3][3];
  fieldPerAmp(point, field);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      gram[i][j] = 0.0;
      for (int channel = 0; channel < 4; channel++) gram[i][j] += field[i][channel] * field[j][channel];
    }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      adjugate[j][i] = gram[i1][j1] * gram[i2][j2] - gram[i1][j2] * gram[i2][j1];
    }
  const double determinant = gram[0][0] * adjugate[0][0] + gram[0][1] * adjugate[1][0] + gram[0][2] * adjugate[2][0];
  if (!(fabs(determinant) > 1e-40)) return false;

  double realWeights[3] = {}, imaginaryWeights[3] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      realWeights[i] += adjugate[i][j] * fieldTesla * orientation[j][0] / determinant;
      imaginaryWeights[i] -= adjugate[i][j] * fieldTesla * orientation[j][1] / determinant;
    }
  for (int channel = 0; channel < 4; channel++) {
    realAmps[channel] = imaginaryAmps[channel] = 0.0;
    for (int axis = 0; axis < 3; axis++) {
      realAmps[channel] += field[axis][channel] * realWeights[axis];
      imaginaryAmps[channel] += field[axis][channel] * imaginaryWeights[axis];
    }
  }
  return true;
}

CoilSettings VectorField::settingsFromCurrents(const double realAmps[4], const double imaginaryAmps[4],
                                               double frequencyHz) const {
  CoilSettings settings = {};
  if (!(frequencyHz > 0.0)) return settings;

  const double angularFrequency = 2.0 * M_PI * frequencyHz;
  const double impedanceOhms = hypot(seriesResistanceOhms, angularFrequency * seriesInductanceHenries -
                                                               1.0 / (angularFrequency * seriesCapacitanceFarads));
  double amps[4], duty[4], largestAmps = 0.0, largestDuty = 0.0;
  for (int channel = 0; channel < 4; channel++) {
    amps[channel] = hypot(realAmps[channel], imaginaryAmps[channel]);
    duty[channel] = 100.0 * amps[channel] * impedanceOhms * M_PI / (4.0 * railVolts) / channelGain[channel];
    largestAmps = fmax(largestAmps, amps[channel]);
    largestDuty = fmax(largestDuty, duty[channel]);
  }
  // b = 0 gives zero currents: zero duty at fraction 1, i.e. the coils go off.
  const double fraction =
      largestAmps > 0.0 ? fmin(1.0, fmin(100.0 / largestDuty, continuousCurrentAmps / largestAmps)) : 1.0;
  for (int channel = 0; channel < 4; channel++) {
    settings.dutyPercent[channel] = (float)(duty[channel] * fraction);
    settings.phaseDegrees[channel] =
        (float)wrapDegrees(phaseOffsetDegrees[channel] - atan2(imaginaryAmps[channel], realAmps[channel]) * degreesPerRadian);
  }
  settings.fieldFraction = (float)fraction;
  return settings;
}
