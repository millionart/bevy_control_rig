use bevy::prelude::Reflect;
use bevy_math::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::math::rig_quat_product;
use crate::{
    hierarchy::{RigTransform, rig_normalize},
    math::source_rotate_vector,
};

#[derive(Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigGeometry {
    #[serde(default = "default_meters_per_unit")]
    #[reflect(default = "default_meters_per_unit")]
    pub meters_per_unit: f64,
    pub near_zero: f64,
    pub normalize_squared: f64,
    pub minimum_length: f32,
    pub solve_near_zero: f64,
    pub basis: [DVec3; 3],
}

fn default_meters_per_unit() -> f64 {
    1.0
}

pub fn source_vector_dot(a: DVec3, b: DVec3) -> f64 {
    (a.x * b.x + a.z * b.z) + a.y * b.y
}

fn safe_normal(value: DVec3, tolerance: f64) -> DVec3 {
    let squared = source_vector_dot(value, value);
    if squared == 1.0 {
        value
    } else if squared < tolerance {
        DVec3::ZERO
    } else {
        value * (1.0 / squared.sqrt())
    }
}

fn normalize(value: DVec3, tolerance: f64) -> Option<DVec3> {
    let squared = source_vector_dot(value, value);
    (squared > tolerance).then(|| value * (1.0 / squared.sqrt()))
}

impl RigGeometry {
    pub fn pole(&self, a: DVec3, b: DVec3, c: DVec3) -> Option<(DVec3, DVec3)> {
        let ab = b - a;
        if ab.abs().max_element() <= self.near_zero {
            return None;
        }
        let Some(ac) = normalize(c - a, self.normalize_squared) else {
            return Some((a, ab * (1.0 / source_vector_dot(ab, ab).sqrt())));
        };
        let projection = a + ac * source_vector_dot(ab, ac);
        normalize(b - projection, self.normalize_squared).map(|pole| (projection, pole))
    }

    fn handedness(&self) -> f64 {
        self.basis[0].dot(self.basis[1].cross(self.basis[2]))
    }

    pub fn between_vectors(&self, a: DVec3, b: DVec3) -> DQuat {
        self.between(
            a,
            b,
            (source_vector_dot(a, a) * source_vector_dot(b, b)).sqrt(),
        )
    }

    fn between(&self, a: DVec3, b: DVec3, norm_ab: f64) -> DQuat {
        let w = norm_ab + source_vector_dot(a, b);
        let (axis, w) = if w >= 1e-6_f32 as f64 * norm_ab {
            (a.cross(b), w)
        } else {
            let magnitudes = self.basis.map(|axis| a.dot(axis).abs());
            let basis = if magnitudes[0] > magnitudes[1] && magnitudes[0] > magnitudes[2] {
                self.basis[1]
            } else {
                -self.basis[0]
            };
            (a.cross(basis), 0.0)
        };
        rig_normalize(DQuat::from_xyzw(axis.x, axis.y, axis.z, w))
    }

    fn positions(
        &self,
        root: DVec3,
        effector: DVec3,
        pole: DVec3,
        mut upper: f64,
        mut lower: f64,
        stretch: bool,
        stretch_start: f64,
        stretch_max: f64,
    ) -> (DVec3, DVec3) {
        let delta = effector - root;
        let mut distance = source_vector_dot(delta, delta).sqrt();
        let centimeters_per_unit = self.meters_per_unit * 100.0;
        let delta_cm = effector * centimeters_per_unit - root * centimeters_per_unit;
        let source_distance =
            source_vector_dot(delta_cm, delta_cm).sqrt() * (0.01 / self.meters_per_unit);
        let direction = if source_distance < self.solve_near_zero {
            distance = self.solve_near_zero;
            self.basis[0]
        } else if distance < self.solve_near_zero {
            distance = source_distance;
            safe_normal(
                delta_cm,
                self.normalize_squared / (0.01 / self.meters_per_unit).powi(2),
            )
        } else {
            safe_normal(delta, self.normalize_squared)
        };
        let pole_delta = pole - root;
        let near_squared = self.solve_near_zero * self.solve_near_zero;
        let bend = if source_vector_dot(pole_delta, pole_delta) < near_squared {
            self.basis[1]
        } else if {
            let plane = direction.cross(pole_delta);
            source_vector_dot(plane, plane) < near_squared
        } {
            let magnitudes = self.basis.map(|axis| direction.dot(axis).abs());
            let basis = if magnitudes[2] > magnitudes[0] && magnitudes[2] > magnitudes[1] {
                self.basis[0]
            } else {
                self.basis[2]
            };
            let first = safe_normal(
                basis - direction * source_vector_dot(basis, direction),
                1e-8_f32 as f64,
            );
            first.cross(direction) * self.handedness()
        } else {
            let bend = pole_delta - direction * source_vector_dot(pole_delta, direction);
            normalize(bend, self.normalize_squared).unwrap_or(bend)
        };
        let mut maximum = upper + lower;
        if stretch {
            let range = stretch_max - stretch_start;
            if range > 1e-4 && maximum > self.solve_near_zero {
                let factor = (stretch_max - 1.0)
                    * ((distance / maximum - stretch_start) / range).clamp(0.0, 1.0);
                if factor > 1e-4 {
                    upper *= 1.0 + factor;
                    lower *= 1.0 + factor;
                    maximum *= 1.0 + factor;
                }
            }
        }
        if distance >= maximum {
            return (root + upper * direction, root + maximum * direction);
        }
        let denominator = 2.0 * upper * distance;
        let cosine = if denominator != 0.0 {
            (upper * upper + distance * distance - lower * lower) / denominator
        } else {
            0.0
        };
        let perpendicular = upper * cosine.clamp(-1.0, 1.0).acos().sin();
        let squared = upper * upper - perpendicular * perpendicular;
        let mut projected = if squared > 0.0 { squared.sqrt() } else { 0.0 };
        if cosine < 0.0 {
            projected = -projected;
        }
        (
            root + projected * direction + perpendicular * bend,
            effector,
        )
    }

    pub fn solve(
        &self,
        bones: &mut [RigTransform; 3],
        pole: DVec3,
        primary: DVec3,
        secondary: DVec3,
        secondary_weight: f32,
        lengths: [impl Into<f64>; 2],
        stretch: bool,
        stretch_start: f32,
        stretch_max: f32,
    ) {
        let root = bones[0].translation;
        let [upper, lower] = lengths.map(Into::into);
        let (joint, effector) = self.positions(
            root,
            bones[2].translation,
            pole,
            upper,
            lower,
            stretch,
            stretch_start as f64,
            stretch_max as f64,
        );
        bones[1].translation = joint;
        bones[2].translation = effector;
        let mut fully_extended = false;
        for index in 0..2 {
            let mut axis = source_rotate_vector(bones[index].rotation, primary);
            let mut target = bones[index + 1].translation - bones[index].translation;
            if target.abs().max_element() <= self.near_zero
                || axis.abs().max_element() <= 1e-4_f32 as f64
            {
                continue;
            }
            target = safe_normal(target, self.normalize_squared);
            if index == 0 {
                let base = safe_normal(effector - root, self.normalize_squared);
                fully_extended = (source_vector_dot(base, target) - 1.0).abs() < 1e-4_f32 as f64;
            }
            bones[index].rotation = rig_normalize(rig_quat_product(
                self.between(axis, target, 1.0),
                bones[index].rotation,
            ));
            if secondary_weight <= 1e-8 {
                continue;
            }
            axis = source_rotate_vector(bones[index].rotation, secondary);
            let mut secondary_target = if fully_extended {
                pole - bones[index].translation
            } else {
                joint - (effector + root) * 0.5
            };
            if secondary_target.abs().max_element() <= self.near_zero
                || axis.abs().max_element() <= 1e-4_f32 as f64
            {
                continue;
            }
            secondary_target -= source_vector_dot(secondary_target, target) * target;
            secondary_target = safe_normal(secondary_target, self.normalize_squared);
            let mut rotation = self.between(axis, secondary_target, 1.0);
            if (secondary_weight - 1.0).abs() > 1e-8 {
                let vector = DVec3::new(rotation.x, rotation.y, rotation.z);
                let axis = if vector.length_squared() >= 1e-8 {
                    vector / vector.length()
                } else {
                    self.basis[0] * self.handedness()
                };
                let angle = (2.0 * rotation.w.clamp(-1.0, 1.0).acos()) as f32;
                rotation =
                    DQuat::from_axis_angle(axis, (angle * secondary_weight.clamp(0.0, 1.0)) as f64);
            }
            bones[index].rotation =
                rig_normalize(rig_quat_product(rotation, bones[index].rotation));
        }
    }
}
