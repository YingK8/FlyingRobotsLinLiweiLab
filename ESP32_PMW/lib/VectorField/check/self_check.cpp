#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "VectorField.h"

static const float spinPhaseDegrees[4] = {270, 90, 180, 0};
static const double identity[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
static const double noRotation[4] = {1, 0, 0, 0};

struct RotatingField {
  double atStart[3], quarterTurnLater[3];

  double length(const double v[3]) const { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
  double startLength() const { return length(atStart); }
  double quarterTurnLength() const { return length(quarterTurnLater); }
  double overlap() const {
    return atStart[0] * quarterTurnLater[0] + atStart[1] * quarterTurnLater[1] + atStart[2] * quarterTurnLater[2];
  }
  void axis(double result[3]) const {
    result[0] = atStart[1] * quarterTurnLater[2] - atStart[2] * quarterTurnLater[1];
    result[1] = atStart[2] * quarterTurnLater[0] - atStart[0] * quarterTurnLater[2];
    result[2] = atStart[0] * quarterTurnLater[1] - atStart[1] * quarterTurnLater[0];
    const double size = length(result);
    for (int i = 0; i < 3; i++) result[i] /= size;
  }
  double semiAxis(bool major) const {
    const double sum = startLength() * startLength() + quarterTurnLength() * quarterTurnLength();
    const double product = startLength() * startLength() * quarterTurnLength() * quarterTurnLength() - overlap() * overlap();
    const double spread = sqrt(fmax(sum * sum - 4 * product, 0));
    return sqrt(fmax(major ? sum + spread : sum - spread, 0) / 2);
  }
};

static RotatingField fieldFrom(const VectorField &coils, const double point[3], const double realAmps[4],
                               const double imaginaryAmps[4]) {
  double field[3][4];
  coils.fieldPerAmp(point, field);
  RotatingField result = {};
  for (int axis = 0; axis < 3; axis++)
    for (int channel = 0; channel < 4; channel++) {
      result.atStart[axis] += field[axis][channel] * realAmps[channel];
      result.quarterTurnLater[axis] -= field[axis][channel] * imaginaryAmps[channel];
    }
  return result;
}

static double wrapToHalfTurn(double degrees) { return fmod(fmod(degrees + 180, 360) + 360, 360) - 180; }

static void checkGeometryMatchesTheGui(const VectorField &coils) {
  const double guiPhases[4] = {0, 270, 180, 90};
  double realAmps[4], imaginaryAmps[4];
  for (int channel = 0; channel < 4; channel++) {
    realAmps[channel] = 1.25 * cos(guiPhases[channel] * M_PI / 180);
    imaginaryAmps[channel] = 1.25 * sin(guiPhases[channel] * M_PI / 180);
  }
  const double heights[2] = {0.0152, 0.0201}, expectedMilliTesla[2] = {4.58, 4.29};
  for (int i = 0; i < 2; i++) {
    const double point[3] = {0, 0, heights[i]};
    const RotatingField field = fieldFrom(coils, point, realAmps, imaginaryAmps);
    assert(fabs(field.semiAxis(true) * 1e3 - expectedMilliTesla[i]) < 0.01);
    assert(fabs(field.semiAxis(false) * 1e3 - expectedMilliTesla[i]) < 0.01);
  }
}

static void checkChannelStrengthsOnAxis(const VectorField &coils) {
  const double point[3] = {0, 0, 0.0201}, expectedMilliTeslaPerAmp[3] = {2.42, 2.42, 0.36};
  double field[3][4];
  coils.fieldPerAmp(point, field);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      double gram = 0;
      for (int channel = 0; channel < 4; channel++) gram += field[i][channel] * field[j][channel];
      if (i == j) assert(fabs(sqrt(gram) * 1e3 - expectedMilliTeslaPerAmp[i]) < 0.01);
      else assert(fabs(gram) < 1e-12 * 2.42e-3 * 2.42e-3);
    }
  for (int axis = 0; axis < 3; axis++)
    assert(fabs(field[axis][0] - field[axis][1] + field[axis][2] - field[axis][3]) < 1e-12);
}

static void checkEveryOrientationIsReached(const VectorField &coils) {
  std::mt19937 generator(0);
  auto uniform = [&](double low, double high) { return std::uniform_real_distribution<double>(low, high)(generator); };
  for (int trial = 0; trial < 200; trial++) {
    const double point[3] = {uniform(-0.005, 0.005), uniform(-0.005, 0.005), uniform(0.010, 0.025)};
    const double quaternion[4] = {uniform(-1, 1), uniform(-0.25, 0.25), uniform(-0.25, 0.25), uniform(-1, 1)};
    const double fieldTesla = uniform(1e-3, 5e-3);
    double orientation[3][3], realAmps[4], imaginaryAmps[4], axis[3];
    assert(VectorField::rotationFromQuaternion(quaternion, orientation));
    assert(coils.currentsFor(point, orientation, fieldTesla, realAmps, imaginaryAmps));
    const RotatingField field = fieldFrom(coils, point, realAmps, imaginaryAmps);
    field.axis(axis);
    const double alignment = axis[0] * orientation[0][2] + axis[1] * orientation[1][2] + axis[2] * orientation[2][2];
    assert(acos(fmin(1.0, alignment)) * 180 / M_PI < 1e-3);
    assert(fabs(field.startLength() / fieldTesla - 1) < 1e-9);
    assert(fabs(field.quarterTurnLength() / fieldTesla - 1) < 1e-9);
    assert(fabs(field.overlap()) < 1e-9 * fieldTesla * fieldTesla);
    for (int i = 0; i < 3; i++) assert(fabs(field.atStart[i] - fieldTesla * orientation[i][0]) < 1e-9 * fieldTesla);
  }
}

static void checkNoRotationGivesTodaysSpin(const VectorField &coils) {
  for (double height : {0.010, 0.015, 0.025}) {
    const double point[3] = {0, 0, height};
    const CoilSettings fromMatrix = coils.settingsFor(point, identity, 2e-3, 100.0);
    const CoilSettings fromQuaternion = coils.settingsFor(point, noRotation, 2e-3, 100.0);
    const double commonShift = wrapToHalfTurn(fromMatrix.phaseDegrees[0] - spinPhaseDegrees[0]);
    for (int channel = 0; channel < 4; channel++) {
      assert(fabs(wrapToHalfTurn(fromMatrix.phaseDegrees[channel] - spinPhaseDegrees[channel] - commonShift)) < 1e-3);
      assert(fabs(fromMatrix.dutyPercent[channel] - fromMatrix.dutyPercent[0]) < 1e-4);
      assert(fromMatrix.dutyPercent[channel] == fromQuaternion.dutyPercent[channel]);
      assert(fromMatrix.phaseDegrees[channel] == fromQuaternion.phaseDegrees[channel]);
    }
  }
}

static void checkTwistOnlyShiftsTheClock(const VectorField &coils) {
  const double point[3] = {0.001, 0.002, 0.016}, twistRadians = 40.0 * M_PI / 180;
  const double tilted[4] = {0.98, 0.1, -0.15, 0.0};
  const double twist[4] = {cos(twistRadians / 2), 0, 0, sin(twistRadians / 2)};
  const double tiltedThenTwisted[4] = {tilted[0] * twist[0] - tilted[3] * twist[3], tilted[1] * twist[0] + tilted[2] * twist[3],
                                       tilted[2] * twist[0] - tilted[1] * twist[3], tilted[3] * twist[0] + tilted[0] * twist[3]};
  const CoilSettings before = coils.settingsFor(point, tilted, 2e-3, 120.0);
  const CoilSettings after = coils.settingsFor(point, tiltedThenTwisted, 2e-3, 120.0);
  for (int channel = 0; channel < 4; channel++) {
    assert(fabs(before.dutyPercent[channel] - after.dutyPercent[channel]) < 1e-3);
    assert(fabs(wrapToHalfTurn(after.phaseDegrees[channel] - before.phaseDegrees[channel] + 40.0)) < 1e-3);
  }
}

static void checkSaturationKeepsTheAxis(const VectorField &coils) {
  const double point[3] = {0.002, -0.001, 0.015}, quaternion[4] = {0.97, 0.05, 0.15, 0.0};
  double orientation[3][3], realAmps[4], imaginaryAmps[4], axis[3];
  VectorField::rotationFromQuaternion(quaternion, orientation);
  coils.currentsFor(point, orientation, 0.5, realAmps, imaginaryAmps);
  const CoilSettings settings = coils.settingsFromCurrents(realAmps, imaginaryAmps, 150.0);
  assert(settings.fieldFraction > 0 && settings.fieldFraction < 1);
  for (int channel = 0; channel < 4; channel++) {
    assert(settings.dutyPercent[channel] <= 100.0f + 1e-4f);
    assert(hypot(realAmps[channel], imaginaryAmps[channel]) * settings.fieldFraction <= 2.0 + 1e-9);
    realAmps[channel] *= settings.fieldFraction;
    imaginaryAmps[channel] *= settings.fieldFraction;
  }
  fieldFrom(coils, point, realAmps, imaginaryAmps).axis(axis);
  for (int i = 0; i < 3; i++) assert(fabs(axis[i] - orientation[i][2]) < 1e-9);
}

static void checkBadRequestsAreSafe(const VectorField &coils) {
  const double point[3] = {0, 0, 0.015}, zeroQuaternion[4] = {0, 0, 0, 0}, tilted[4] = {0.98, 0.1, -0.15, 0.0};
  const double stretched[3][3] = {{2, 0, 0}, {0, 1, 0}, {0, 0, 1}}, mirrored[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}};
  const CoilSettings noField[4] = {coils.settingsFor(point, stretched, 2e-3, 100), coils.settingsFor(point, mirrored, 2e-3, 100),
                                   coils.settingsFor(point, zeroQuaternion, 2e-3, 100),
                                   coils.settingsFor(point, identity, 2e-3, 0.0)};
  for (const CoilSettings &settings : noField) {
    assert(settings.fieldFraction == 0);
    for (int channel = 0; channel < 4; channel++) assert(settings.dutyPercent[channel] == 0);
  }

  const CoilSettings off = coils.settingsFor(point, identity, 0.0, 100);
  assert(off.fieldFraction == 1);
  for (int channel = 0; channel < 4; channel++) assert(off.dutyPercent[channel] == 0);

  const CoilSettings positive = coils.settingsFor(point, tilted, 2e-3, 120.0);
  const CoilSettings negative = coils.settingsFor(point, tilted, -2e-3, 120.0);
  assert(negative.fieldFraction == positive.fieldFraction);
  for (int channel = 0; channel < 4; channel++) {
    assert(fabs(negative.dutyPercent[channel] - positive.dutyPercent[channel]) < 1e-3);
    assert(fabs(wrapToHalfTurn(negative.phaseDegrees[channel] - positive.phaseDegrees[channel] - 180.0)) < 1e-3);
  }
}

int main() {
  const VectorField coils(spinPhaseDegrees);
  checkGeometryMatchesTheGui(coils);
  checkChannelStrengthsOnAxis(coils);
  checkEveryOrientationIsReached(coils);
  checkNoRotationGivesTodaysSpin(coils);
  checkTwistOnlyShiftsTheClock(coils);
  checkSaturationKeepsTheAxis(coils);
  checkBadRequestsAreSafe(coils);
  printf("VectorField ok: phase offsets %.1f %.1f %.1f %.1f deg\n", coils.phaseOffsetDegrees[0],
         coils.phaseOffsetDegrees[1], coils.phaseOffsetDegrees[2], coils.phaseOffsetDegrees[3]);
}
