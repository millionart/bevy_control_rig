use bevy::{
    math::{DMat4, DQuat, DVec3},
    prelude::Reflect,
};
use serde::{Deserialize, Serialize};

use crate::{
    math::source_rotate_vector,
    math::{RigValue, rig_quat_product},
};

#[derive(Reflect, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigTransform {
    pub rotation: DQuat,
    pub translation: DVec3,
    pub scale: DVec3,
}

impl Default for RigTransform {
    fn default() -> Self {
        Self {
            rotation: DQuat::IDENTITY,
            translation: DVec3::ZERO,
            scale: DVec3::ONE,
        }
    }
}

pub fn rig_normalize(rotation: DQuat) -> DQuat {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{
            vdupq_n_f64, vgetq_lane_f64, vmulq_f64, vrsqrteq_f64, vrsqrtsq_f64,
        };

        let squared = rotation.y.mul_add(rotation.y, rotation.x * rotation.x)
            + rotation.w.mul_add(rotation.w, rotation.z * rotation.z);
        if squared >= 1e-8_f32 as f64 {
            let reciprocal = unsafe {
                let value = vdupq_n_f64(squared);
                let estimate = vrsqrteq_f64(value);
                let estimate =
                    vmulq_f64(vrsqrtsq_f64(value, vmulq_f64(estimate, estimate)), estimate);
                let estimate =
                    vmulq_f64(vrsqrtsq_f64(value, vmulq_f64(estimate, estimate)), estimate);
                vgetq_lane_f64::<0>(estimate)
            };
            rotation * reciprocal
        } else {
            DQuat::IDENTITY
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if rotation.length_squared() >= 1e-8_f32 as f64 {
            rotation.normalize()
        } else {
            DQuat::IDENTITY
        }
    }
}

pub fn rig_slerp(a: DQuat, b: DQuat, weight: f64) -> DQuat {
    let dot = ((a.x * b.x + a.z * b.z) + a.y * b.y) + a.w * b.w;
    let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
    let mut a_weight = 1.0 - weight;
    let mut b_weight = weight * sign;
    if dot * sign < 0.9999 {
        let angle = (dot * sign).acos();
        let inverse_sine = 1.0 / angle.sin();
        a_weight = (a_weight * angle).sin() * inverse_sine;
        b_weight = (b_weight * angle).sin() * inverse_sine;
    }
    rig_normalize(a * a_weight + b * b_weight)
}

pub fn rig_quat_near(a: DQuat, b: DQuat, tolerance: f64) -> bool {
    (a - b)
        .to_array()
        .iter()
        .all(|value| value.abs() <= tolerance)
        || (a + b)
            .to_array()
            .iter()
            .all(|value| value.abs() <= tolerance)
}

impl RigTransform {
    fn scale_reciprocal(self) -> DVec3 {
        DVec3::from_array(self.scale.to_array().map(|value| {
            if value.abs() <= 1e-8_f32 as f64 {
                0.0
            } else {
                value.recip()
            }
        }))
    }

    pub fn transform_point(self, point: DVec3) -> DVec3 {
        source_rotate_vector(self.rotation, self.scale * point) + self.translation
    }

    pub fn inverse(self) -> Self {
        if self.scale.abs().max_element() <= 1e-8_f32 as f64 {
            return Self::default();
        }
        let scale = self.scale_reciprocal();
        let rotation = self.rotation.conjugate();
        Self {
            rotation,
            translation: -source_rotate_vector(rotation, scale * self.translation),
            scale,
        }
    }

    pub fn inverse_transform_point(self, point: DVec3) -> DVec3 {
        self.inverse_transform_vector(point - self.translation)
    }

    pub fn inverse_transform_vector(self, vector: DVec3) -> DVec3 {
        source_rotate_vector(self.rotation.conjugate(), vector) * self.scale_reciprocal()
    }

    pub fn value(self) -> RigValue {
        RigValue::List(vec![
            RigValue::Quaternion(self.rotation),
            RigValue::Vector(self.translation),
            RigValue::Vector(self.scale),
        ])
    }

    pub fn from_value(value: &RigValue) -> Result<Self, String> {
        if let RigValue::List(fields) = value
            && let [
                RigValue::Quaternion(rotation),
                RigValue::Vector(translation),
                RigValue::Vector(scale),
            ] = fields.as_slice()
        {
            return Ok(Self {
                rotation: *rotation,
                translation: *translation,
                scale: *scale,
            });
        }
        Err("rig transform value is invalid".to_owned())
    }

    fn matrix(self) -> DMat4 {
        DMat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    fn from_matrix(matrix: DMat4, scale: DVec3) -> Self {
        let axes = [
            matrix.x_axis.truncate(),
            matrix.y_axis.truncate(),
            matrix.z_axis.truncate(),
        ];
        let axes = std::array::from_fn::<_, 3, _>(|index| {
            let axis = axes[index];
            let length_squared = axis.length_squared();
            let axis = if length_squared > 0.0 {
                axis / length_squared.sqrt()
            } else {
                axis
            };
            axis * if scale[index] >= 0.0 { 1.0 } else { -1.0 }
        });
        let rotation = if axes
            .iter()
            .any(|axis| axis.abs().max_element() <= 1e-4_f32 as f64)
        {
            DQuat::IDENTITY
        } else {
            let diagonal = DVec3::new(axes[0].x, axes[1].y, axes[2].z);
            let trace = diagonal.element_sum();
            let rotation = if trace > 0.0 {
                let root = (trace + 1.0).sqrt();
                let factor = 0.5 / root;
                DQuat::from_xyzw(
                    (axes[1].z - axes[2].y) * factor,
                    (axes[2].x - axes[0].z) * factor,
                    (axes[0].y - axes[1].x) * factor,
                    0.5 * root,
                )
            } else {
                let mut i = if diagonal.y > diagonal.x { 1 } else { 0 };
                if diagonal.z > diagonal[i] {
                    i = 2;
                }
                let j = (i + 1) % 3;
                let k = (j + 1) % 3;
                let root = (diagonal[i] - diagonal[j] - diagonal[k] + 1.0).sqrt();
                let factor = 0.5 / root;
                let mut value = [0.0; 4];
                value[i] = 0.5 * root;
                value[3] = (axes[j][k] - axes[k][j]) * factor;
                value[j] = (axes[i][j] + axes[j][i]) * factor;
                value[k] = (axes[i][k] + axes[k][i]) * factor;
                DQuat::from_array(value)
            };
            rig_normalize(rotation)
        };
        Self {
            rotation,
            translation: matrix.w_axis.truncate(),
            scale,
        }
    }

    pub fn global(self, parent: Self) -> Self {
        let scale = self.scale * parent.scale;
        if self.scale.min_element() < 0.0 || parent.scale.min_element() < 0.0 {
            return Self::from_matrix(parent.matrix() * self.matrix(), scale);
        }
        Self {
            rotation: rig_normalize(rig_quat_product(parent.rotation, self.rotation)),
            translation: source_rotate_vector(parent.rotation, parent.scale * self.translation)
                + parent.translation,
            scale,
        }
    }

    pub fn relative(self, parent: Self) -> Self {
        if (1.0 - parent.rotation.length_squared()).abs() > 0.01_f32 as f64 {
            return Self::default();
        }
        let reciprocal = parent.scale_reciprocal();
        let scale = self.scale * reciprocal;
        if self.scale.min_element() < 0.0 || parent.scale.min_element() < 0.0 {
            let matrix = parent.matrix();
            let zero_axes = [matrix.x_axis, matrix.y_axis, matrix.z_axis]
                .iter()
                .all(|axis| axis.truncate().abs().max_element() <= 1e-8_f32 as f64);
            let inverse = if zero_axes || matrix.determinant() == 0.0 {
                DMat4::IDENTITY
            } else {
                matrix.inverse()
            };
            return Self::from_matrix(inverse * self.matrix(), scale);
        }
        let inverse_rotation = parent.rotation.conjugate();
        Self {
            rotation: rig_normalize(rig_quat_product(inverse_rotation, self.rotation)),
            translation: source_rotate_vector(
                inverse_rotation,
                self.translation - parent.translation,
            ) * reciprocal,
            scale,
        }
    }

    fn lerp(self, target: Self, weight: f32) -> Self {
        let weight = weight as f64;
        Self {
            rotation: rig_slerp(self.rotation, target.rotation, weight),
            translation: self.translation + weight * (target.translation - self.translation),
            scale: self.scale + weight * (target.scale - self.scale),
        }
    }

    pub fn same(self, other: Self, translation_tolerance: f64) -> bool {
        let tolerance = 1e-4_f32 as f64;
        (self.translation - other.translation).abs().max_element() <= translation_tolerance
            && (self.scale - other.scale).abs().max_element() <= tolerance
            && rig_quat_near(self.rotation, other.rotation, tolerance)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigSpace {
    Local,
    Global,
}

impl RigSpace {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "LocalSpace" => Some(Self::Local),
            "GlobalSpace" => Some(Self::Global),
            _ => None,
        }
    }
}

#[derive(Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigBone {
    pub name: String,
    pub parent: Option<usize>,
    pub initial_local: RigTransform,
    pub initial_global: RigTransform,
}

#[derive(Clone, Copy, Debug)]
struct BonePose {
    local: RigTransform,
    global: RigTransform,
    dirty: Option<RigSpace>,
}

#[derive(Clone, Debug)]
pub struct RigHierarchy {
    bones: Vec<RigBone>,
    initial: Vec<BonePose>,
    current: Vec<BonePose>,
    translation_tolerance: f64,
}

impl RigHierarchy {
    pub fn new(bones: Vec<RigBone>, translation_tolerance: f64) -> Result<Self, String> {
        for (index, bone) in bones.iter().enumerate() {
            if bone.parent.is_some_and(|parent| parent >= index)
                || bones[..index]
                    .iter()
                    .any(|previous| previous.name.eq_ignore_ascii_case(&bone.name))
            {
                return Err(
                    "rig bones must have unique names and precede their children".to_owned(),
                );
            }
        }
        if !translation_tolerance.is_finite() || translation_tolerance < 0.0 {
            return Err("rig translation tolerance is invalid".to_owned());
        }
        let initial: Vec<_> = bones
            .iter()
            .map(|bone| BonePose {
                local: bone.initial_local,
                global: bone.initial_global,
                dirty: None,
            })
            .collect();
        Ok(Self {
            bones,
            current: initial.clone(),
            initial,
            translation_tolerance,
        })
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.bones
            .iter()
            .position(|bone| bone.name.eq_ignore_ascii_case(name))
    }

    pub fn parent(&self, index: usize) -> Option<usize> {
        self.bones.get(index).and_then(|bone| bone.parent)
    }

    pub fn set_input_pose(&mut self, local: &[RigTransform]) -> Result<(), String> {
        if local.len() != self.bones.len() {
            return Err("rig input pose size does not match the hierarchy".to_owned());
        }
        for (pose, &transform) in self.current.iter_mut().zip(local) {
            pose.local = transform;
            pose.dirty = Some(RigSpace::Global);
        }
        Ok(())
    }

    fn pose(&mut self, index: usize, initial: bool) -> Result<&mut BonePose, String> {
        if initial {
            &mut self.initial
        } else {
            &mut self.current
        }
        .get_mut(index)
        .ok_or_else(|| "rig bone index is invalid".to_owned())
    }

    pub fn transform(
        &mut self,
        index: usize,
        space: RigSpace,
        initial: bool,
    ) -> Result<RigTransform, String> {
        let pose = *self.pose(index, initial)?;
        if pose.dirty == Some(space) {
            let parent = if let Some(parent) = self.bones[index].parent {
                self.transform(parent, RigSpace::Global, initial)?
            } else {
                RigTransform::default()
            };
            let transform = match space {
                RigSpace::Local => {
                    let mut local = pose.global.relative(parent);
                    if local.scale.abs().min_element() <= 1e-8
                        && parent.scale.abs().min_element() <= 1e-8
                    {
                        local.translation = pose.local.translation;
                        local.scale = pose.local.scale;
                    }
                    local
                }
                RigSpace::Global => pose.local.global(parent),
            };
            let pose = self.pose(index, initial)?;
            match space {
                RigSpace::Local => pose.local = transform,
                RigSpace::Global => pose.global = transform,
            }
            pose.dirty = None;
            return Ok(transform);
        }
        Ok(match space {
            RigSpace::Local => pose.local,
            RigSpace::Global => pose.global,
        })
    }

    fn dirty_children(
        &mut self,
        index: usize,
        initial: bool,
        propagate: bool,
    ) -> Result<(), String> {
        let (clean, dirty) = if propagate {
            (RigSpace::Local, RigSpace::Global)
        } else {
            (RigSpace::Global, RigSpace::Local)
        };
        for child in index + 1..self.bones.len() {
            if self.bones[child].parent != Some(index)
                || self.pose(child, initial)?.dirty == Some(dirty)
            {
                continue;
            }
            self.transform(child, clean, initial)?;
            if propagate {
                self.dirty_children(child, initial, propagate)?;
            }
            self.pose(child, initial)?.dirty = Some(dirty);
        }
        Ok(())
    }

    pub fn set_transform(
        &mut self,
        index: usize,
        space: RigSpace,
        initial: bool,
        value: RigTransform,
        propagate: bool,
    ) -> Result<(), String> {
        let pose = *self.pose(index, initial)?;
        let previous = match space {
            RigSpace::Local => pose.local,
            RigSpace::Global => pose.global,
        };
        if pose.dirty != Some(space) && previous.same(value, self.translation_tolerance) {
            return Ok(());
        }
        self.transform(index, space, initial)?;
        self.dirty_children(index, initial, propagate)?;
        let pose = self.pose(index, initial)?;
        match space {
            RigSpace::Local => {
                pose.local = value;
                pose.dirty = Some(RigSpace::Global);
            }
            RigSpace::Global => {
                pose.global = value;
                pose.dirty = Some(RigSpace::Local);
            }
        }
        Ok(())
    }

    pub fn item_index(&self, item: &RigValue) -> Result<Option<usize>, String> {
        if let RigValue::List(fields) = item
            && let [RigValue::Name(kind), RigValue::Name(name)] = fields.as_slice()
        {
            return match kind.as_str() {
                "Bone" => Ok(self.index(name)),
                "None" => Ok(None),
                _ => Err(format!("rig transform element type is unsupported: {kind}")),
            };
        }
        Err("rig element key is invalid".to_owned())
    }

    pub fn call_transform_unit(
        &mut self,
        function: &str,
        arguments: &mut [RigValue],
        construction: bool,
    ) -> Result<(), String> {
        use RigValue::*;
        match (function, arguments) {
            (
                "FRigUnit_GetTransform::Execute",
                [item, Name(space), Bool(initial), output, _cached],
            ) => {
                if let Some(index) = self.item_index(item)?
                    && let Some(space) = RigSpace::from_name(space)
                {
                    *output = self.transform(index, space, *initial)?.value();
                }
            }
            (
                "FRigUnit_SetTranslation::Execute"
                | "FRigUnit_SetRotation::Execute"
                | "FRigUnit_SetScale::Execute",
                [
                    item,
                    Name(space),
                    Bool(initial),
                    value,
                    F32(weight),
                    Bool(propagate),
                    _cached,
                ],
            ) => {
                if *weight < 1e-8 {
                    return Ok(());
                }
                let Some(index) = self.item_index(item)? else {
                    return Ok(());
                };
                let Some(space) = RigSpace::from_name(space) else {
                    return Ok(());
                };
                let previous = self.transform(index, space, *initial)?;
                let mut transform = previous;
                match (function, value) {
                    ("FRigUnit_SetTranslation::Execute", Vector(value)) => {
                        transform.translation = *value
                    }
                    ("FRigUnit_SetRotation::Execute", Quaternion(value)) => {
                        transform.rotation = *value
                    }
                    ("FRigUnit_SetScale::Execute", Vector(value)) => transform.scale = *value,
                    _ => return Err("rig transform unit value type is invalid".to_owned()),
                }
                if *weight < 1.0 {
                    transform = previous.lerp(transform, *weight);
                }
                self.set_transform(index, space, *initial, transform, *propagate)?;
                if *initial && construction {
                    self.set_transform(index, space, false, transform, *propagate)?;
                }
            }
            _ => {
                return Err(format!(
                    "rig transform unit signature is unsupported: {function}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{RigInstruction, RigProgram, RigValueRef};

    fn transform(translation: DVec3) -> RigTransform {
        RigTransform {
            translation,
            ..Default::default()
        }
    }

    fn hierarchy() -> RigHierarchy {
        RigHierarchy::new(
            vec![
                RigBone {
                    name: "Parent".to_owned(),
                    parent: None,
                    initial_local: transform(DVec3::X),
                    initial_global: transform(DVec3::X),
                },
                RigBone {
                    name: "Child".to_owned(),
                    parent: Some(0),
                    initial_local: transform(DVec3::Y),
                    initial_global: transform(DVec3::X + DVec3::Y),
                },
                RigBone {
                    name: "Tip".to_owned(),
                    parent: Some(1),
                    initial_local: transform(DVec3::Z),
                    initial_global: transform(DVec3::ONE),
                },
                RigBone {
                    name: "Other".to_owned(),
                    parent: None,
                    initial_local: transform(DVec3::NEG_X),
                    initial_global: transform(DVec3::NEG_X),
                },
            ],
            1e-4_f32 as f64 * 0.01,
        )
        .unwrap()
    }

    fn assert_transform(actual: RigTransform, expected: RigTransform) {
        assert!(
            (actual.translation - expected.translation)
                .abs()
                .max_element()
                < 1e-9,
            "{actual:?} != {expected:?}"
        );
        assert!(
            (actual.scale - expected.scale).abs().max_element() < 1e-9,
            "{actual:?} != {expected:?}"
        );
        assert!(
            actual.rotation.dot(expected.rotation).abs() > 1.0 - 1e-9,
            "{actual:?} != {expected:?}"
        );
    }

    fn item(name: &str) -> RigValue {
        RigValue::List(vec![
            RigValue::Name("Bone".to_owned()),
            RigValue::Name(name.to_owned()),
        ])
    }

    #[test]
    fn control_rig_hierarchy_preserves_source_spaces_propagation_and_input_lifecycle() {
        use RigSpace::*;
        let mut rig = hierarchy();
        let moved = RigTransform {
            translation: DVec3::new(2.0, 3.0, 4.0),
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            scale: DVec3::new(2.0, 3.0, 4.0),
        };
        rig.set_transform(0, Global, false, moved, true).unwrap();
        assert_transform(
            rig.transform(1, Global, false).unwrap(),
            RigTransform {
                translation: DVec3::new(-1.0, 3.0, 4.0),
                ..moved
            },
        );
        assert_transform(
            rig.transform(2, Global, false).unwrap(),
            RigTransform {
                translation: DVec3::new(-1.0, 3.0, 8.0),
                ..moved
            },
        );
        assert_transform(rig.transform(0, Global, true).unwrap(), transform(DVec3::X));
        assert_transform(
            rig.transform(3, Global, false).unwrap(),
            transform(DVec3::NEG_X),
        );
        let old_tip = rig.transform(2, Global, false).unwrap();
        let old_child = rig.transform(1, Global, false).unwrap();
        let next = RigTransform {
            translation: moved.translation + DVec3::X,
            ..moved
        };
        rig.set_transform(0, Local, false, next, false).unwrap();
        assert_transform(rig.transform(1, Global, false).unwrap(), old_child);
        assert_transform(rig.transform(2, Global, false).unwrap(), old_tip);
        assert_transform(
            rig.transform(1, Local, false).unwrap(),
            transform(DVec3::new(0.0, 4.0 / 3.0, 0.0)),
        );

        let input = [
            transform(DVec3::X),
            transform(DVec3::Y),
            transform(DVec3::Z),
            transform(DVec3::NEG_X),
        ];
        for scale in [2.0, 1.0, 3.0, 1.0] {
            let mut frame = input;
            frame[0].scale = DVec3::splat(scale);
            rig.set_input_pose(&frame).unwrap();
            let before = rig.transform(0, Local, false).unwrap();
            rig.set_transform(
                0,
                Local,
                false,
                RigTransform {
                    scale: before.scale + DVec3::new(0.4, 0.0, 0.4),
                    ..before
                },
                true,
            )
            .unwrap();
            assert_transform(
                rig.transform(2, Global, false).unwrap(),
                RigTransform {
                    translation: DVec3::new(1.0, scale, scale + 0.4),
                    scale: DVec3::new(scale + 0.4, scale, scale + 0.4),
                    ..Default::default()
                },
            );
        }

        rig.set_input_pose(&input).unwrap();
        rig.set_transform(
            0,
            Local,
            false,
            RigTransform {
                scale: DVec3::ZERO,
                ..input[0]
            },
            true,
        )
        .unwrap();
        rig.transform(1, Global, false).unwrap();
        rig.set_transform(
            1,
            Global,
            false,
            RigTransform {
                rotation: DQuat::from_rotation_x(0.3),
                ..transform(DVec3::X)
            },
            false,
        )
        .unwrap();
        assert_transform(
            rig.transform(1, Local, false).unwrap(),
            RigTransform {
                rotation: DQuat::from_rotation_x(0.3),
                ..input[1]
            },
        );

        rig.set_input_pose(&input).unwrap();
        rig.transform(0, Local, false).unwrap();
        rig.set_transform(
            0,
            Local,
            false,
            transform(DVec3::new(1.0 + 0.5e-6, 0.0, 0.0)),
            true,
        )
        .unwrap();
        assert_eq!(rig.transform(0, Local, false).unwrap(), input[0]);
        rig.set_transform(
            0,
            Local,
            false,
            transform(DVec3::new(1.0 + 2e-6, 0.0, 0.0)),
            true,
        )
        .unwrap();
        assert_eq!(
            rig.transform(0, Local, false).unwrap().translation.x,
            1.0 + 2e-6
        );
        assert!(rig.set_input_pose(&[]).is_err());
        assert!(rig.transform(9, Global, false).is_err());
        let mut bad = rig.bones.clone();
        bad[0].parent = Some(1);
        assert!(RigHierarchy::new(bad, 1e-6).is_err());
        let mut bad = rig.bones.clone();
        bad[1].name = "PARENT".to_owned();
        assert!(RigHierarchy::new(bad, 1e-6).is_err());
    }

    #[test]
    fn control_rig_transform_units_execute_through_vm_with_source_weights_and_construction() {
        use RigSpace::*;
        use RigValue::*;
        let mut rig = hierarchy();
        let mut arguments = vec![
            item("parent"),
            Name("LocalSpace".to_owned()),
            Bool(false),
            Quaternion(DQuat::from_rotation_x(2.0)),
            F32(0.25),
            Bool(true),
            List(vec![]),
        ];
        let call = RigInstruction::Call {
            function: "FRigUnit_SetRotation::Execute".to_owned(),
            arguments: (0..7)
                .map(|register| RigValueRef {
                    register,
                    path: vec![],
                })
                .collect(),
            outputs: vec![6],
        };
        let program = RigProgram {
            initial_values: arguments.clone(),
            literal_count: 6,
            entries: [("Forwards Solve".to_owned(), 0)].into(),
            instructions: vec![call, RigInstruction::Exit],
        };
        program
            .execute("Forwards Solve", &mut arguments, |_, function, values| {
                rig.call_transform_unit(function, values, false)
            })
            .unwrap();
        assert_transform(
            rig.transform(0, Local, false).unwrap(),
            RigTransform {
                rotation: DQuat::from_rotation_x(0.5),
                ..transform(DVec3::X)
            },
        );
        assert_transform(rig.transform(0, Local, true).unwrap(), transform(DVec3::X));

        arguments[2] = Bool(true);
        arguments[3] = Vector(DVec3::splat(3.0));
        arguments[4] = F32(1.5);
        rig.call_transform_unit("FRigUnit_SetScale::Execute", &mut arguments, false)
            .unwrap();
        assert_eq!(
            rig.transform(0, Local, true).unwrap().scale,
            DVec3::splat(3.0)
        );
        assert_eq!(rig.transform(0, Local, false).unwrap().scale, DVec3::ONE);
        arguments[3] = Vector(DVec3::splat(5.0));
        rig.call_transform_unit("FRigUnit_SetScale::Execute", &mut arguments, true)
            .unwrap();
        assert_eq!(
            rig.transform(0, Local, false).unwrap().scale,
            DVec3::splat(5.0)
        );
        arguments[2] = Bool(false);
        arguments[3] = Vector(DVec3::splat(9.0));
        arguments[4] = F32(0.5e-8);
        rig.call_transform_unit("FRigUnit_SetScale::Execute", &mut arguments, false)
            .unwrap();
        assert_eq!(
            rig.transform(0, Local, false).unwrap().scale,
            DVec3::splat(5.0)
        );

        arguments[0] = item("child");
        arguments[1] = Name("GlobalSpace".to_owned());
        arguments[3] = Vector(DVec3::new(3.0, 4.0, 5.0));
        arguments[4] = F32(1.0);
        rig.call_transform_unit("FRigUnit_SetTranslation::Execute", &mut arguments, false)
            .unwrap();
        let mut get = vec![
            item("Child"),
            Name("GlobalSpace".to_owned()),
            Bool(false),
            RigTransform::default().value(),
            List(vec![]),
        ];
        rig.call_transform_unit("FRigUnit_GetTransform::Execute", &mut get, false)
            .unwrap();
        assert_eq!(
            RigTransform::from_value(&get[3]).unwrap().translation,
            DVec3::new(3.0, 4.0, 5.0)
        );
        let output = get[3].clone();
        get[0] = item("Missing");
        rig.call_transform_unit("FRigUnit_GetTransform::Execute", &mut get, false)
            .unwrap();
        assert_eq!(get[3], output);
        get[0] = item("Child");
        get[1] = Name("InvalidSpace".to_owned());
        rig.call_transform_unit("FRigUnit_GetTransform::Execute", &mut get, false)
            .unwrap();
        assert_eq!(get[3], output);
        assert!(
            rig.call_transform_unit("Missing", &mut arguments, false)
                .is_err()
        );
        arguments[3] = Bool(false);
        assert!(
            rig.call_transform_unit("FRigUnit_SetRotation::Execute", &mut arguments, false)
                .is_err()
        );
        let encoded = serde_json::to_string(&rig.bones).unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<RigBone>>(&encoded).unwrap(),
            rig.bones
        );
    }

    #[test]
    fn rig_transform_preserves_up_for_near_unit_yaw() {
        let transform = RigTransform {
            rotation: DQuat::from_xyzw(0.0, 0.382_683_455_944_061_3, 0.0, 0.923_879_504_203_796_4),
            ..Default::default()
        };
        assert_eq!(transform.inverse_transform_vector(DVec3::Y), DVec3::Y);
        assert_eq!(transform.transform_point(DVec3::Y), DVec3::Y);
        let child = RigTransform {
            translation: DVec3::Y,
            ..Default::default()
        };
        assert_eq!(child.global(transform).translation, DVec3::Y);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn rig_normalization_matches_native_arm64_quaternions() {
        for (input, expected) in [
            (
                DQuat::IDENTITY,
                DQuat::from_xyzw(0.0, 0.0, 0.0, 0.999_999_999_950_951_3),
            ),
            (
                DQuat::from_xyzw(
                    0.073_906_391_859_054_57,
                    0.073_906_391_859_054_57,
                    0.703_233_897_686_004_6,
                    0.703_233_897_686_004_6,
                ),
                DQuat::from_xyzw(
                    0.073_906_386_710_638_58,
                    0.073_906_386_710_638_58,
                    0.703_233_848_697_811,
                    0.703_233_848_697_811,
                ),
            ),
            (
                DQuat::from_xyzw(
                    -0.703_233_957_290_649_4,
                    0.703_233_838_081_359_9,
                    -0.073_906_227_946_281_43,
                    0.073_906_272_649_765_01,
                ),
                DQuat::from_xyzw(
                    -0.703_233_923_017_270_8,
                    0.703_233_803_807_987_1,
                    -0.073_906_224_344_327_77,
                    0.073_906_269_047_809_17,
                ),
            ),
            (
                DQuat::from_xyzw(
                    0.082_567_840_814_590_45,
                    0.078_878_626_227_378_85,
                    0.702_785_491_943_359_4,
                    0.702_177_524_566_650_4,
                ),
                DQuat::from_xyzw(
                    0.082_567_840_409_883_48,
                    0.078_878_625_840_754_58,
                    0.702_785_488_498_650_3,
                    0.702_177_521_124_921_3,
                ),
            ),
        ] {
            assert_eq!(rig_normalize(input), expected);
        }
        assert_eq!(
            RigTransform::default()
                .global(RigTransform::default())
                .rotation
                .w,
            0.999_999_999_950_951_3
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn rig_slerp_preserves_source_partial_weight_scalar_order() {
        for (before, solved, unnormalized) in [
            (
                DQuat::from_xyzw(
                    0.633_735_685_482_602,
                    -0.736_975_225_023_681_4,
                    0.193_413_863_622_010_02,
                    -0.133_557_762_439_721_62,
                ),
                DQuat::from_xyzw(
                    0.631_203_251_261_262,
                    -0.728_302_773_287_365_3,
                    0.231_574_036_973_998_34,
                    -0.132_404_650_291_106_93,
                ),
                DQuat::from_xyzw(
                    0.633_193_895_569_690_7,
                    -0.734_912_962_656_805_3,
                    0.202_984_118_571_771_62,
                    -0.133_288_692_261_303_35,
                ),
            ),
            (
                DQuat::from_xyzw(
                    0.695_247_581_071_448,
                    -0.695_145_745_365_060_7,
                    0.129_272_117_010_432_8,
                    -0.129_197_187_975_249_87,
                ),
                DQuat::from_xyzw(
                    0.703_233_957_290_649_4,
                    -0.703_233_838_081_359_9,
                    0.073_906_227_946_281_43,
                    -0.073_906_272_649_765_01,
                ),
                DQuat::from_xyzw(
                    0.697_654_066_685_290_4,
                    -0.697_577_625_220_709_1,
                    0.115_492_958_476_517_05,
                    -0.115_436_747_242_431_15,
                ),
            ),
        ] {
            let expected = rig_normalize(unnormalized);
            assert_eq!(rig_slerp(before, solved, 0.25), expected);
            assert_eq!(rig_slerp(before, -solved, 0.25), expected);
        }
    }

    #[test]
    fn control_rig_transform_math_preserves_source_scale_and_rotation_branches() {
        let parent = RigTransform {
            rotation: DQuat::from_rotation_z(0.7),
            translation: DVec3::new(1.0, 2.0, 3.0),
            scale: DVec3::new(2.0, 3.0, 4.0),
        };
        let local = RigTransform {
            rotation: DQuat::from_rotation_y(0.2),
            translation: DVec3::new(0.3, 0.5, 0.7),
            scale: DVec3::new(1.3, 1.2, 0.9),
        };
        assert_transform(local.global(parent).relative(parent), local);
        let mirrored = RigTransform {
            scale: DVec3::new(-2.0, 2.0, 2.0),
            ..parent
        };
        assert_transform(
            local.global(mirrored),
            RigTransform {
                rotation: parent.rotation * DQuat::from_rotation_y(-0.2),
                translation: parent.rotation * DVec3::new(-0.6, 1.0, 1.4) + parent.translation,
                scale: DVec3::new(-2.6, 2.4, 1.8),
            },
        );
        assert_transform(local.global(mirrored).relative(mirrored), local);
        let sheared = RigTransform {
            rotation: DQuat::from_rotation_y(0.7),
            ..Default::default()
        }
        .global(RigTransform {
            rotation: DQuat::from_rotation_x(1.6),
            scale: DVec3::new(-2.0, 3.0, 4.0),
            ..Default::default()
        });
        assert_transform(
            sheared,
            RigTransform {
                rotation: DQuat::from_xyzw(
                    0.7434454231671356,
                    -0.1405078490594962,
                    -0.33268607902903363,
                    0.5629088913357118,
                ),
                scale: DVec3::new(-2.0, 3.0, 4.0),
                ..Default::default()
            },
        );
        let tiny = RigTransform {
            scale: DVec3::new(-2e-10, 2e-10, 2e-10),
            ..mirrored
        };
        let tiny_global = local.global(tiny);
        assert!(
            tiny_global
                .rotation
                .dot(parent.rotation * DQuat::from_rotation_y(-0.2))
                .abs()
                > 1.0 - 1e-12
        );
        assert_eq!(local.relative(tiny).translation, local.translation);
        let zero = RigTransform {
            scale: DVec3::ZERO,
            ..parent
        };
        assert_eq!(local.relative(zero).scale, DVec3::ZERO);
        let invalid_rotation = RigTransform {
            rotation: DQuat::from_xyzw(0.0, 0.0, 0.0, 2.0),
            ..parent
        };
        assert_eq!(local.relative(invalid_rotation), RigTransform::default());
        let value = RigTransform::default().value();
        assert_eq!(
            RigTransform::from_value(&value).unwrap(),
            RigTransform::default()
        );
        assert!(RigTransform::from_value(&RigValue::List(vec![])).is_err());
        assert_eq!(
            rig_normalize(DQuat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
            DQuat::IDENTITY
        );
        for angle in [0.001, 0.02, 2.0, -2.0] {
            let target = DQuat::from_rotation_y(angle);
            let result = rig_slerp(DQuat::IDENTITY, -target, 0.25);
            let expected = if angle.abs() < 0.03 {
                rig_normalize(DQuat::IDENTITY * 0.75 + target * 0.25)
            } else {
                DQuat::from_rotation_y(angle * 0.25)
            };
            let cosine = result.dot(expected).abs()
                / (result.length_squared() * expected.length_squared()).sqrt();
            assert!(cosine > 1.0 - 1e-14);
        }
    }
}
