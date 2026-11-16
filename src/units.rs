use bevy::platform::collections::HashMap;
use bevy_math::DVec3;

use crate::{
    hierarchy::{RigHierarchy, RigSpace, RigTransform, rig_slerp},
    math::{RigValue, rig_spring_damper, source_rotate_vector},
    rig_ik::{RigGeometry, source_vector_dot},
};

pub type RigLineTrace<'a> =
    dyn FnMut(DVec3, DVec3, &str, bool) -> Result<Option<(DVec3, DVec3)>, String> + 'a;

pub struct RigUnitContext<'a, CurveName = String> {
    pub hierarchy: &'a mut RigHierarchy,
    pub curves: &'a HashMap<CurveName, f32>,
    pub game_world: bool,
    pub construction: bool,
    pub delta_seconds: f64,
    pub geometry: &'a RigGeometry,
    pub to_world: RigTransform,
    pub line_trace: Option<&'a mut RigLineTrace<'a>>,
}

pub fn branch_label(condition: bool, label: &mut String) {
    *label = if label.is_empty() {
        if condition { "True" } else { "False" }
    } else {
        "Completed"
    }
    .to_owned();
}

pub fn length_slot(value: &mut RigValue, meters_per_unit: f64) -> Result<(&mut f32, f64), String> {
    match value {
        RigValue::F32(value) => Ok((value, 1.0)),
        RigValue::Centimeters(value) => Ok((value, 100.0 * meters_per_unit)),
        _ => Err("rig length type is invalid".to_owned()),
    }
}

pub fn length_in_units(value: &mut RigValue, meters_per_unit: f64) -> Result<f64, String> {
    let (length, factor) = length_slot(value, meters_per_unit)?;
    Ok(f64::from(*length) * (1.0 / factor))
}

impl<CurveName: AsRef<str>> RigUnitContext<'_, CurveName> {
    pub fn call(&mut self, function: &str, arguments: &mut [RigValue]) -> Result<(), String> {
        use RigValue::*;
        let meters_per_unit = self.geometry.meters_per_unit;
        match (function, arguments) {
            (
                "FRigUnit_TwoBoneIKSimplePerItem::Execute",
                [
                    a,
                    b,
                    end,
                    effector,
                    Vector(primary),
                    Vector(secondary),
                    F32(secondary_weight),
                    Vector(pole),
                    Name(pole_kind),
                    pole_space,
                    Bool(stretch),
                    F32(stretch_start),
                    F32(stretch_max),
                    F32(weight),
                    length_a,
                    length_b,
                    Bool(propagate),
                    _debug,
                    _cached_a,
                    _cached_b,
                    _cached_end,
                    _cached_pole,
                ],
            ) => {
                let Some(a) = self.hierarchy.item_index(a)? else {
                    return Ok(());
                };
                let Some(b) = self.hierarchy.item_index(b)? else {
                    return Ok(());
                };
                let end = self.hierarchy.item_index(end)?;
                let pole_space = self.hierarchy.item_index(pole_space)?;
                if *weight <= 1e-8 {
                    return Ok(());
                }
                let mut pole = *pole;
                if let Some(space) = pole_space {
                    let transform = self.hierarchy.transform(space, RigSpace::Global, false)?;
                    pole = source_rotate_vector(transform.rotation, pole);
                    if pole_kind != "Direction" {
                        pole += transform.translation;
                    }
                }
                let transform_a = self.hierarchy.transform(a, RigSpace::Global, false)?;
                let transform_b = self.hierarchy.transform(b, RigSpace::Global, false)?;
                let mut bones = [
                    transform_a,
                    RigTransform {
                        translation: transform_b.translation,
                        ..transform_a
                    },
                    RigTransform::from_value(effector)?,
                ];
                let minimum_length = f64::from(self.geometry.minimum_length);
                let mut lengths = [
                    length_in_units(length_a, meters_per_unit)?,
                    length_in_units(length_b, meters_per_unit)?,
                ];
                let centimeters_per_unit = 100.0 * meters_per_unit;
                for (index, (parent, child)) in [(a, Some(b)), (b, end)].into_iter().enumerate() {
                    if lengths[index] < minimum_length
                        && let Some(child) = child
                    {
                        let initial = self.hierarchy.transform(parent, RigSpace::Global, true)?;
                        let scale =
                            if source_vector_dot(initial.scale, initial.scale) > 1e-8_f32 as f64 {
                                bones[index].scale / initial.scale
                            } else {
                                bevy::math::DVec3::ONE
                            };
                        let child = self.hierarchy.transform(child, RigSpace::Global, true)?;
                        let delta_cm = (initial.translation * centimeters_per_unit
                            - child.translation * centimeters_per_unit)
                            * scale;
                        let length_cm = source_vector_dot(delta_cm, delta_cm).sqrt() as f32;
                        lengths[index] = f64::from(length_cm) * (0.01 / meters_per_unit);
                    }
                }
                if lengths.iter().any(|&length| length < minimum_length) {
                    return Ok(());
                }
                self.geometry.solve(
                    &mut bones,
                    pole,
                    *primary,
                    *secondary,
                    *secondary_weight,
                    lengths,
                    *stretch,
                    *stretch_start,
                    *stretch_max,
                );
                if *weight < 1.0 {
                    let joint = bones[0].inverse_transform_point(bones[1].translation);
                    let effector = bones[1].inverse_transform_point(bones[2].translation);
                    bones[0].rotation =
                        rig_slerp(transform_a.rotation, bones[0].rotation, *weight as f64);
                    bones[1].rotation =
                        rig_slerp(transform_b.rotation, bones[1].rotation, *weight as f64);
                    if let Some(end) = end {
                        let current = self.hierarchy.transform(end, RigSpace::Global, false)?;
                        bones[2].rotation =
                            rig_slerp(current.rotation, bones[2].rotation, *weight as f64);
                    }
                    bones[1].translation = bones[0].transform_point(joint);
                    bones[2].translation = bones[1].transform_point(effector);
                }
                for (index, transform) in [Some(a), Some(b), end].into_iter().zip(bones) {
                    if let Some(index) = index {
                        self.hierarchy.set_transform(
                            index,
                            RigSpace::Global,
                            false,
                            transform,
                            *propagate,
                        )?;
                    }
                }
            }
            (
                "FRigUnit_SpringInterpV2::Execute",
                [
                    F32(target) | Centimeters(target),
                    F32(strength),
                    F32(damping),
                    F32(force) | Centimeters(force),
                    Bool(use_current),
                    F32(current) | Centimeters(current),
                    F32(target_velocity_amount),
                    Bool(initialize_from_target),
                    F32(result) | Centimeters(result),
                    F32(velocity) | Centimeters(velocity),
                    F32(simulated) | Centimeters(simulated),
                    List(state),
                ],
            ) => {
                let [
                    F32(previous_target) | Centimeters(previous_target),
                    F32(state_velocity) | Centimeters(state_velocity),
                    Bool(previous_valid),
                ] = state.as_mut_slice()
                else {
                    return Err("rig float spring state layout is invalid".to_owned());
                };
                let angular_frequency = *strength * 2.0 * std::f32::consts::PI;
                let stiffness = angular_frequency * angular_frequency;
                let mut adjusted_target = *target;
                if stiffness.abs() > 1e-8 {
                    adjusted_target += *force / stiffness;
                } else {
                    *state_velocity =
                        (*state_velocity as f64 + *force as f64 * self.delta_seconds) as f32;
                }
                let mut value = if *use_current { *current } else { *simulated };
                if (!*use_current || *initialize_from_target) && !*previous_valid {
                    value = adjusted_target;
                }
                let delta_seconds = self.delta_seconds as f32;
                if delta_seconds > 1e-8 {
                    let target_velocity = if *previous_valid {
                        (adjusted_target - *previous_target)
                            * (*target_velocity_amount / delta_seconds)
                    } else {
                        0.0
                    };
                    rig_spring_damper(
                        &mut value,
                        state_velocity,
                        adjusted_target,
                        target_velocity,
                        delta_seconds,
                        stiffness.sqrt() / (2.0 * std::f32::consts::PI),
                        *damping,
                    );
                    *previous_target = adjusted_target;
                    *previous_valid = true;
                }
                *simulated = value;
                *result = value;
                *velocity = *state_velocity;
            }
            ("FRigUnit_BeginExecution::Execute", []) => self.construction = false,
            ("FRigUnit_PrepareForExecution::Execute", []) => self.construction = true,
            ("FRigVMFunction_ControlFlowBranch::Execute", [Bool(condition), Name(label)]) => {
                branch_label(*condition, label)
            }
            (
                "DISPATCH_RigVMDispatch_If::Condition:bool,True:FRigElementKey,False:FRigElementKey,Result:FRigElementKey",
                [Bool(condition), List(a), List(b), List(output)],
            ) => {
                if a.len() != 2
                    || b.len() != 2
                    || !a
                        .iter()
                        .chain(b.iter())
                        .all(|field| matches!(field, Name(_)))
                {
                    return Err("rig selection element key is invalid".to_owned());
                }
                *output = if *condition { a.clone() } else { b.clone() };
            }
            (
                "FRigUnit_GetCurveValue::Execute",
                [Name(name), Bool(valid), F32(output), _cached],
            ) => {
                let value = self.curves.iter().find_map(|(key, value)| {
                    key.as_ref().eq_ignore_ascii_case(name).then_some(*value)
                });
                *valid = value.is_some();
                *output = value.unwrap_or(0.0);
            }
            (function, arguments) => {
                self.hierarchy
                    .call_transform_unit(function, arguments, self.construction)?
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::{RigBone, RigTransform, rig_quat_near};
    use bevy::math::DVec3;
    use bevy::platform::collections::HashMap;
    use bevy_math::DQuat;

    fn geometry() -> RigGeometry {
        RigGeometry {
            meters_per_unit: 1.0,
            near_zero: 1e-4_f32 as f64 * 0.01,
            normalize_squared: 1e-8_f32 as f64 * 0.0001,
            minimum_length: 1e-8_f32 * 0.01,
            solve_near_zero: 1e-4 * 0.01,
            basis: [DVec3::X, DVec3::Z, DVec3::Y],
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn control_rig_pole_and_ik_preserve_source_vector_order() {
        let geometry = geometry();
        let (projection, direction) = geometry
            .pole(
                DVec3::ZERO,
                DVec3::new(0.5, 2.0, 1.0),
                DVec3::new(3.0, 5.0, 4.0),
            )
            .unwrap();
        assert_eq!(projection, DVec3::new(0.93, 1.55, 1.24));
        assert_eq!(
            direction,
            DVec3::new(
                -0.644_597_252_536_441,
                0.674_578_520_096_275_3,
                -0.359_775_210_718_013_55
            )
        );

        let (projection, direction) = geometry
            .pole(
                DVec3::new(
                    8.925_669_401_805_589,
                    94.925_253_710_169_7,
                    -1.717_870_670_449_023_2,
                ),
                DVec3::new(
                    15.873_860_534_693_453,
                    53.219_259_420_003_425,
                    3.023_558_198_420_145_6,
                ),
                DVec3::new(
                    18.975_329_039_394_797,
                    13.409_921_254_359_077,
                    -1.599_313_650_113_666_4,
                ),
            )
            .unwrap();
        assert_eq!(
            projection,
            DVec3::new(
                14.095_286_683_713_653,
                52.993_179_729_490_16,
                -1.656_884_085_497_952
            )
        );
        assert_eq!(
            direction,
            DVec3::new(
                0.354_857_215_564_911_07,
                0.045_106_931_841_553_17,
                0.933_831_741_408_140_3
            )
        );

        let mut bones = [RigTransform::default(); 3];
        bones[0].translation = DVec3::new(
            -9.017_389_605_395_621,
            94.903_783_166_794_78,
            -3.242_183_316_710_252_5,
        );
        bones[2].translation = DVec3::new(
            -17.076_301_574_707_03,
            13.465_195_894_241_333,
            -5.072_212_219_238_281,
        );
        geometry.solve(
            &mut bones,
            DVec3::new(
                -35.484_576_782_270_3,
                52.930_200_444_555,
                33.857_362_981_432_14,
            ),
            DVec3::X,
            DVec3::Y,
            0.0,
            [42.572_25_f32, 40.196_82_f32],
            false,
            0.75,
            1.25,
        );
        assert_eq!(
            bones[1].translation,
            DVec3::new(
                -16.246_864_416_758_697,
                53.175_769_941_310_58,
                1.105_702_083_001_855_1
            )
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn control_rig_ik_uses_source_rotated_axes_for_recorded_input() {
        let mut bones = [RigTransform::default(); 3];
        bones[0].translation = DVec3::new(
            -0.107_983_336_080_343_25,
            0.820_029_764_262_862_6,
            0.012_280_340_777_750_365,
        );
        bones[2].translation = DVec3::new(
            -0.017_576_217_651_367_188,
            0.184_527_194_127_440_45,
            -0.431_320_190_429_687_5,
        );
        let initial = DQuat::from_xyzw(
            0.680_567_360_775_579_1,
            -0.702_881_658_872_854_5,
            0.100_461_295_035_963_95,
            -0.180_811_972_133_496_28,
        );
        bones[0].rotation = initial;
        bones[1].rotation = initial;
        geometry().solve(
            &mut bones,
            DVec3::new(
                -0.265_311_954_503_788_14,
                0.251_595_211_436_327_24,
                0.390_121_188_974_275_5,
            ),
            DVec3::X,
            DVec3::NEG_Z,
            1.0,
            [0.425_722_5_f32, 0.401_968_18_f32],
            false,
            0.75,
            1.25,
        );
        assert_eq!(bones[0].rotation.w as f32, -0.210_394_19);
    }

    fn solve_ik_unit_positions(
        hierarchy: &mut RigHierarchy,
        root: RigTransform,
        effector: DVec3,
        pole: DVec3,
        lengths: [f32; 2],
    ) -> [DVec3; 2] {
        use RigValue::*;

        hierarchy
            .set_transform(0, RigSpace::Global, false, root, true)
            .unwrap();
        let curves = HashMap::default();
        let mut context = RigUnitContext::<String> {
            hierarchy,
            curves: &curves,
            game_world: true,
            construction: false,
            delta_seconds: 0.0,
            geometry: &geometry(),
            to_world: RigTransform::default(),
            line_trace: None,
        };
        let effector = RigTransform {
            translation: effector,
            ..Default::default()
        };
        let mut args = [
            item("Bone0"),
            item("Bone1"),
            item("Bone2"),
            effector.value(),
            Vector(DVec3::ZERO),
            Vector(DVec3::ZERO),
            F32(0.0),
            Vector(pole),
            Name("Location".to_owned()),
            item(""),
            Bool(false),
            F32(0.75),
            F32(1.25),
            F32(1.0),
            F32(lengths[0]),
            F32(lengths[1]),
            Bool(true),
            List(vec![]),
            List(vec![]),
            List(vec![]),
            List(vec![]),
            List(vec![]),
        ];
        context
            .call("FRigUnit_TwoBoneIKSimplePerItem::Execute", &mut args)
            .unwrap();
        [1, 2].map(|bone| {
            context
                .hierarchy
                .transform(bone, RigSpace::Global, false)
                .unwrap()
                .translation
        })
    }

    #[test]
    fn control_rig_ik_uses_the_native_coincident_goal_boundary() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }

        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            root: [f64; 3],
            effector: [f64; 3],
            pole: [f64; 3],
            joint: [f64; 3],
            end: [f64; 3],
        }

        let fixture: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/goal.replay.json")).unwrap();
        for case in fixture.cases {
            let mut hierarchy = hierarchy();
            let root = RigTransform {
                translation: DVec3::from_array(case.root),
                ..Default::default()
            };
            let actual = solve_ik_unit_positions(
                &mut hierarchy,
                root,
                DVec3::from_array(case.effector),
                DVec3::from_array(case.pole),
                [0.5, 0.75],
            );
            for (position, expected) in actual.into_iter().zip([case.joint, case.end]) {
                assert!(
                    position.distance(DVec3::from_array(expected)) < 1e-12,
                    "{}: {position:?}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn control_rig_ik_rounds_automatic_lengths_in_source_centimeters() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }

        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            root: [f64; 3],
            scale: [f64; 3],
            effector: [f64; 3],
            pole: [f64; 3],
            initials: [[f64; 3]; 3],
            initial_scales: [[f64; 3]; 3],
            joint: [f64; 3],
            end: [f64; 3],
        }

        let fixture: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/lengths.replay.json")).unwrap();
        for case in fixture.cases {
            let globals: [RigTransform; 3] = std::array::from_fn(|index| RigTransform {
                translation: DVec3::from_array(case.initials[index]),
                scale: DVec3::from_array(case.initial_scales[index]),
                ..Default::default()
            });
            let bones = globals
                .iter()
                .copied()
                .enumerate()
                .map(|(index, initial_global)| {
                    let parent = index.checked_sub(1);
                    RigBone {
                        name: format!("Bone{index}"),
                        parent,
                        initial_local: parent.map_or(initial_global, |parent| {
                            initial_global.relative(globals[parent])
                        }),
                        initial_global,
                    }
                })
                .collect();
            let mut hierarchy = RigHierarchy::new(bones, 1e-6).unwrap();
            let root = RigTransform {
                translation: DVec3::from_array(case.root),
                scale: DVec3::from_array(case.scale),
                ..Default::default()
            };
            let actual = solve_ik_unit_positions(
                &mut hierarchy,
                root,
                DVec3::from_array(case.effector),
                DVec3::from_array(case.pole),
                [0.0, 0.0],
            );
            for (position, expected) in actual.into_iter().zip([case.joint, case.end]) {
                assert!(
                    position.distance(DVec3::from_array(expected)) < 1e-8,
                    "{}: {position:?}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn control_rig_ik_geometry_preserves_native_degenerate_axes_stretch_and_roll() {
        let geometry = geometry();
        let mut threshold_case = geometry.clone();
        threshold_case.normalize_squared = 1.0;
        assert!(
            threshold_case
                .pole(DVec3::ZERO, DVec3::new(1.0, 1.0, 0.0), DVec3::X * 2.0)
                .is_none()
        );
        assert!(
            geometry
                .pole(DVec3::ZERO, DVec3::splat(0.8e-6), DVec3::Y)
                .is_none()
        );
        assert!(
            geometry
                .pole(DVec3::ZERO, DVec3::X, DVec3::X * 2.0)
                .is_none()
        );
        assert_eq!(
            geometry.pole(DVec3::ZERO, DVec3::Y, DVec3::X * 1e-8),
            Some((DVec3::ZERO, DVec3::Y))
        );
        assert!(
            geometry
                .pole(DVec3::ZERO, DVec3::X + DVec3::Y * 0.9e-6, DVec3::X * 2.0)
                .is_none()
        );
        assert!(
            geometry
                .pole(DVec3::ZERO, DVec3::X + DVec3::Y * 1.1e-6, DVec3::X * 2.0)
                .is_some()
        );
        let cases = [
            (
                DVec3::X * 3.0,
                DVec3::Y,
                [1.0, 1.0],
                false,
                DVec3::X,
                DVec3::X * 2.0,
            ),
            (
                DVec3::X * 3.0,
                DVec3::Y,
                [1.0, 1.0],
                true,
                DVec3::X * 1.25,
                DVec3::X * 2.5,
            ),
            (
                DVec3::X * 2.2,
                DVec3::Y,
                [1.0, 1.0],
                true,
                DVec3::new(1.1, (1.175_f64.powi(2) - 1.1_f64.powi(2)).sqrt(), 0.0),
                DVec3::X * 2.2,
            ),
            (
                DVec3::X * 0.5,
                DVec3::Y,
                [0.5, 2.0],
                false,
                DVec3::NEG_X * 0.5,
                DVec3::X * 0.5,
            ),
            (
                DVec3::X,
                DVec3::ZERO,
                [1.0, 1.0],
                false,
                DVec3::new(0.5, 0.0, 3.0_f64.sqrt() * 0.5),
                DVec3::X,
            ),
            (
                DVec3::X,
                DVec3::X * 2.0,
                [1.0, 1.0],
                false,
                DVec3::new(0.5, 0.0, 3.0_f64.sqrt() * 0.5),
                DVec3::X,
            ),
            (
                DVec3::Y,
                DVec3::Y * 2.0,
                [1.0, 1.0],
                false,
                DVec3::new(0.0, 0.5, -3.0_f64.sqrt() * 0.5),
                DVec3::Y,
            ),
        ];
        for (target, pole, lengths, stretch, joint, end) in cases {
            let offset = DVec3::new(5.0, -3.0, 7.0);
            let mut bones = [RigTransform {
                translation: offset,
                ..Default::default()
            }; 3];
            bones[2].translation += target;
            geometry.solve(
                &mut bones,
                pole + offset,
                DVec3::ZERO,
                DVec3::ZERO,
                0.0,
                lengths,
                stretch,
                0.75,
                1.25,
            );
            assert!(
                bones[1].translation.distance(joint + offset) < 1e-12,
                "{target:?}: {:?}",
                bones[1]
            );
            assert!(bones[2].translation.distance(end + offset) < 1e-12);
        }
        let mut bones = [RigTransform::default(); 3];
        geometry.solve(
            &mut bones,
            DVec3::Y,
            DVec3::ZERO,
            DVec3::ZERO,
            0.0,
            [1.0, 1.0],
            false,
            0.75,
            1.25,
        );
        assert!(bones[1].translation.x > 0.0 && bones[1].translation.x < 1e-6);
        assert!((bones[1].translation.y - 1.0).abs() < 1e-12);
        assert_eq!(bones[2].translation, DVec3::ZERO);
        for weight in [0.0, 0.25, 1.0, 2.0] {
            let mut bones = [RigTransform::default(); 3];
            bones[2].translation = DVec3::X * 3.0;
            geometry.solve(
                &mut bones,
                DVec3::Z,
                DVec3::X,
                DVec3::Y,
                weight,
                [1.0, 1.0],
                false,
                0.75,
                1.25,
            );
            let angle = if weight == 1.0 {
                std::f64::consts::FRAC_PI_2
            } else {
                (std::f32::consts::FRAC_PI_2 * weight.clamp(0.0, 1.0)) as f64
            };
            for bone in &bones[..2] {
                assert!(
                    rig_quat_near(bone.rotation, DQuat::from_rotation_x(angle), 1e-12),
                    "{weight}: {:?}",
                    bone.rotation
                );
            }
        }
        let mut bones = [RigTransform::default(); 3];
        bones[2].translation = DVec3::NEG_X * 3.0;
        geometry.solve(
            &mut bones,
            DVec3::Z,
            DVec3::X,
            DVec3::Y,
            0.0,
            [1.0, 1.0],
            false,
            0.75,
            1.25,
        );
        assert!(rig_quat_near(
            bones[0].rotation,
            DQuat::from_xyzw(0.0, -1.0, 0.0, 0.0),
            1e-12
        ));
        let mut bones = [RigTransform::default(); 3];
        bones[2].translation = DVec3::Y * 3.0;
        geometry.solve(
            &mut bones,
            DVec3::Z,
            DVec3::X * 2.0,
            DVec3::Y,
            0.0,
            [1.0, 1.0],
            false,
            0.75,
            1.25,
        );
        assert!(rig_quat_near(
            bones[0].rotation,
            DQuat::from_xyzw(0.0, 0.0, 2.0, 1.0).normalize(),
            1e-12
        ));
    }

    fn item(name: &str) -> RigValue {
        RigValue::List(vec![
            RigValue::Name("Bone".to_owned()),
            RigValue::Name(name.to_owned()),
        ])
    }

    fn hierarchy() -> RigHierarchy {
        let positions = [
            DVec3::ZERO,
            DVec3::X,
            DVec3::X + DVec3::Y,
            DVec3::new(1.0, 1.0, 1.0),
        ];
        RigHierarchy::new(
            (0..4)
                .map(|index| RigBone {
                    name: format!("Bone{index}"),
                    parent: (index > 0).then(|| index - 1),
                    initial_local: RigTransform {
                        translation: positions[index]
                            - if index > 0 {
                                positions[index - 1]
                            } else {
                                DVec3::ZERO
                            },
                        ..Default::default()
                    },
                    initial_global: RigTransform {
                        translation: positions[index],
                        ..Default::default()
                    },
                })
                .collect(),
            1e-6,
        )
        .unwrap()
    }
}
