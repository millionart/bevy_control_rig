use bevy::math::{Quat, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoneSolve {
    pub hip_rotation_world: Quat,
    pub knee_rotation_world: Quat,
    pub knee_position: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicTwoBoneSolve {
    pub hip_rotation_world: Quat,
    pub knee_rotation_world: Quat,
    pub knee_position: Vec3,
    pub effector_position: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoleVector {
    pub projection: Vec3,
    pub direction: Vec3,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StableTwoBonePole {
    reference_local: Vec3,
    last_world: Vec3,
    initialized: bool,
}

const MIN_DISTANCE: f32 = 1e-4;

fn clamp_reach(distance: f32, upper_len: f32, lower_len: f32) -> f32 {
    let max = (upper_len + lower_len).max(2.0 * MIN_DISTANCE) - 1e-3;
    let min = ((upper_len - lower_len).abs()).max(MIN_DISTANCE);
    distance.clamp(min, max.max(min))
}

fn perpendicular_fallback(axis: Vec3, pole: Vec3) -> Vec3 {
    let candidate = pole - axis * pole.dot(axis);
    if candidate.length_squared() > 1e-8 {
        candidate.normalize()
    } else {
        let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        axis.cross(helper).normalize()
    }
}

fn source_fallback_bend_direction(axis: Vec3) -> Vec3 {
    let reference = if axis.y.abs() > axis.x.abs() && axis.y.abs() > axis.z.abs() {
        Vec3::X
    } else {
        Vec3::Y
    };
    let first = (reference - axis * reference.dot(axis)).normalize_or_zero();
    axis.cross(first).normalize_or_zero()
}

fn rotation_between_normals(from: Vec3, to: Vec3) -> Option<Quat> {
    let from = from.try_normalize()?;
    let to = to.try_normalize()?;
    Some(Quat::from_rotation_arc(from, to).normalize())
}

fn weighted_rotation(rotation: Quat, weight: f32) -> Quat {
    Quat::IDENTITY.slerp(rotation, weight.clamp(0.0, 1.0))
}

fn align_secondary_axis(
    rotation: Quat,
    secondary_axis: Vec3,
    target_axis: Vec3,
    target: Vec3,
    weight: f32,
) -> Quat {
    if weight <= MIN_DISTANCE {
        return rotation;
    }
    let axis = rotation * secondary_axis;
    let target = target - target_axis * target.dot(target_axis);
    let Some(rotation_delta) = rotation_between_normals(axis, target) else {
        return rotation;
    };
    (weighted_rotation(rotation_delta, weight) * rotation).normalize()
}

pub fn solve_basic_two_bone_ik(
    hip: Vec3,
    hip_rotation: Quat,
    _knee: Vec3,
    target: Vec3,
    pole: Vec3,
    upper_len: f32,
    lower_len: f32,
    primary_axis: Vec3,
    secondary_axis: Vec3,
    secondary_axis_weight: f32,
) -> Option<BasicTwoBoneSolve> {
    if upper_len <= MIN_DISTANCE || lower_len <= MIN_DISTANCE {
        return None;
    }
    let desired_delta = target - hip;
    let desired_length = desired_delta.length();
    let desired_dir = if desired_length <= MIN_DISTANCE {
        Vec3::X
    } else {
        desired_delta / desired_length
    };
    let pole_delta = pole - hip;
    let bend_dir = if pole_delta.length_squared() <= MIN_DISTANCE * MIN_DISTANCE {
        Vec3::Z
    } else {
        let projected = pole_delta - desired_dir * pole_delta.dot(desired_dir);
        projected
            .try_normalize()
            .unwrap_or_else(|| source_fallback_bend_direction(desired_dir))
    };
    let max_length = upper_len + lower_len;
    let (knee_position, effector_position) = if desired_length >= max_length {
        (
            hip + desired_dir * upper_len,
            hip + desired_dir * max_length,
        )
    } else {
        let denominator = 2.0 * upper_len * desired_length.max(MIN_DISTANCE);
        let cosine = (upper_len * upper_len + desired_length * desired_length
            - lower_len * lower_len)
            / denominator;
        let reverse_upper = cosine < 0.0;
        let angle = cosine.clamp(-1.0, 1.0).acos();
        let joint_line_distance = upper_len * angle.sin();
        let mut projected_distance = (upper_len * upper_len
            - joint_line_distance * joint_line_distance)
            .max(0.0)
            .sqrt();
        if reverse_upper {
            projected_distance = -projected_distance;
        }
        (
            hip + desired_dir * projected_distance + bend_dir * joint_line_distance,
            target,
        )
    };

    let upper_target_axis = (knee_position - hip).try_normalize()?;
    let fully_extended = (effector_position - hip)
        .try_normalize()
        .is_some_and(|axis| (axis.dot(upper_target_axis) - 1.0).abs() < 1e-4);
    let upper_primary_axis = hip_rotation * primary_axis;
    let upper_primary_rotation = rotation_between_normals(upper_primary_axis, upper_target_axis)?;
    let mut solved_hip_rotation = (upper_primary_rotation * hip_rotation).normalize();
    let upper_secondary_target = if fully_extended {
        pole - hip
    } else {
        knee_position - (effector_position + hip) * 0.5
    };
    solved_hip_rotation = align_secondary_axis(
        solved_hip_rotation,
        secondary_axis,
        upper_target_axis,
        upper_secondary_target,
        secondary_axis_weight,
    );

    let lower_target_axis = (effector_position - knee_position).try_normalize()?;
    let lower_primary_axis = hip_rotation * primary_axis;
    let lower_primary_rotation = rotation_between_normals(lower_primary_axis, lower_target_axis)?;
    let mut solved_knee_rotation = (lower_primary_rotation * hip_rotation).normalize();
    let lower_secondary_target = if fully_extended {
        pole - knee_position
    } else {
        knee_position - (effector_position + hip) * 0.5
    };
    solved_knee_rotation = align_secondary_axis(
        solved_knee_rotation,
        secondary_axis,
        lower_target_axis,
        lower_secondary_target,
        secondary_axis_weight,
    );

    Some(BasicTwoBoneSolve {
        hip_rotation_world: solved_hip_rotation,
        knee_rotation_world: solved_knee_rotation,
        knee_position,
        effector_position,
    })
}

pub fn scaled_initial_bone_length(
    initial_delta: Vec3,
    initial_scale: Vec3,
    current_scale: Vec3,
) -> f32 {
    let ratio = if initial_scale.length_squared() > 1e-8 {
        current_scale / initial_scale
    } else {
        Vec3::ONE
    };
    (initial_delta * ratio).length()
}

pub fn calculate_pole_vector(hip: Vec3, knee: Vec3, ankle: Vec3) -> Option<PoleVector> {
    let hip_to_knee = knee - hip;
    if hip_to_knee.length_squared() <= 1e-8 {
        return None;
    }
    let hip_to_ankle = ankle - hip;
    let Some(axis) = hip_to_ankle.try_normalize() else {
        return Some(PoleVector {
            projection: hip,
            direction: hip_to_knee.normalize(),
        });
    };
    let projection = hip + axis * hip_to_knee.dot(axis);
    let direction = (knee - projection).try_normalize()?;
    Some(PoleVector {
        projection,
        direction,
    })
}

fn projected_direction(direction: Vec3, axis: Vec3) -> Option<Vec3> {
    (direction - axis * direction.dot(axis)).try_normalize()
}

impl StableTwoBonePole {
    pub fn resolve(
        &mut self,
        armature_rotation: Quat,
        hip: Vec3,
        knee: Vec3,
        ankle: Vec3,
        target: Vec3,
        fallback_world: Vec3,
    ) -> Vec3 {
        let axis = (target - hip)
            .try_normalize()
            .or_else(|| (ankle - hip).try_normalize())
            .unwrap_or(Vec3::Y);
        let current = calculate_pole_vector(hip, knee, ankle).map(|pole| pole.direction);
        if !self.initialized {
            let reference_world = current
                .and_then(|direction| projected_direction(direction, axis))
                .or_else(|| projected_direction(fallback_world, axis))
                .unwrap_or_else(|| perpendicular_fallback(axis, fallback_world));
            self.reference_local = armature_rotation.inverse() * reference_world;
            self.last_world = reference_world;
            self.initialized = true;
        }
        let reference_world = projected_direction(armature_rotation * self.reference_local, axis);
        let last_world = projected_direction(self.last_world, axis);
        let current_world = current.and_then(|direction| projected_direction(direction, axis));
        let fallback_world = projected_direction(fallback_world, axis);
        let mut resolved = reference_world
            .or(last_world)
            .or(current_world)
            .or(fallback_world)
            .unwrap_or_else(|| perpendicular_fallback(axis, Vec3::Y));
        if let Some(last_world) = last_world
            && resolved.dot(last_world) < 0.0
        {
            resolved = -resolved;
        }
        self.last_world = resolved;
        resolved
    }

    pub fn last_direction_world(&self) -> Vec3 {
        self.last_world
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

pub fn solve_two_bone_ik(
    hip: Vec3,
    knee: Vec3,
    ankle: Vec3,
    target: Vec3,
    upper_len: f32,
    lower_len: f32,
    pole: Vec3,
) -> Option<TwoBoneSolve> {
    if upper_len <= MIN_DISTANCE || lower_len <= MIN_DISTANCE {
        return None;
    }

    let to_target = target - hip;
    let raw_distance = to_target.length();
    if raw_distance <= MIN_DISTANCE || !raw_distance.is_finite() {
        return None;
    }
    let axis = to_target / raw_distance;
    let distance = clamp_reach(raw_distance, upper_len, lower_len);
    let reach_point = hip + axis * distance;

    let cos_hip = (distance * distance + upper_len * upper_len - lower_len * lower_len)
        / (2.0 * distance * upper_len);
    let sin_hip = (1.0 - cos_hip * cos_hip).clamp(0.0, 1.0).sqrt();
    let side = perpendicular_fallback(axis, pole);

    let upper_dir = axis * cos_hip + side * sin_hip;
    let knee_position = hip + upper_dir * upper_len;

    let cur_upper_dir = (knee - hip).normalize_or_zero();
    if cur_upper_dir.length_squared() < 0.5 {
        return None;
    }
    let hip_rotation_world = Quat::from_rotation_arc(cur_upper_dir, upper_dir.normalize());

    let cur_lower_dir = (ankle - knee).normalize_or_zero();
    let new_lower_dir = (reach_point - knee_position).normalize_or_zero();
    if cur_lower_dir.length_squared() < 0.5 || new_lower_dir.length_squared() < 0.5 {
        return None;
    }
    let knee_rotation_world = Quat::from_rotation_arc(cur_lower_dir, new_lower_dir);

    Some(TwoBoneSolve {
        hip_rotation_world,
        knee_rotation_world,
        knee_position,
    })
}

pub fn world_rotation_to_local(parent_world: Quat, world_rotation: Quat) -> Quat {
    parent_world.inverse() * world_rotation
}

pub fn blend_ik_rotation(current: Quat, target: Quat, weight: f32) -> Quat {
    current.slerp(target, weight.clamp(0.0, 1.0))
}

pub fn child_world_rotation_after_parent_rotation(
    parent_rotation_delta: Quat,
    child_world_rotation: Quat,
    current_child_axis_world: Vec3,
    target_child_axis_world: Vec3,
) -> Option<Quat> {
    let current_after_parent =
        (parent_rotation_delta * current_child_axis_world).try_normalize()?;
    let target_child_axis_world = target_child_axis_world.try_normalize()?;
    let correction = Quat::from_rotation_arc(current_after_parent, target_child_axis_world);
    Some((correction * parent_rotation_delta * child_world_rotation).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_bone_length_uses_the_initial_global_scale_ratio() {
        let delta = Vec3::new(3.0, 4.0, 0.0);
        assert!(
            (scaled_initial_bone_length(delta, Vec3::new(1.0, 2.0, 1.0), Vec3::new(2.0, 1.0, 1.0))
                - 40.0_f32.sqrt())
            .abs()
                < 1e-6
        );
        assert_eq!(scaled_initial_bone_length(delta, Vec3::ONE, Vec3::ONE), 5.0);
        assert_eq!(
            scaled_initial_bone_length(delta, Vec3::ZERO, Vec3::splat(2.0)),
            5.0
        );
    }

    const HIP: Vec3 = Vec3::ZERO;
    const UPPER: f32 = 0.45;
    const LOWER: f32 = 0.45;

    fn straight_chain() -> (Vec3, Vec3) {
        (
            Vec3::new(0.0, -UPPER, 0.0),
            Vec3::new(0.0, -(UPPER + LOWER), 0.0),
        )
    }

    #[test]
    fn reachable_target_is_hit_exactly() {
        let (knee, ankle) = straight_chain();
        let target = Vec3::new(0.15, -0.80, 0.05);
        let solve = solve_two_bone_ik(HIP, knee, ankle, target, UPPER, LOWER, Vec3::Z).unwrap();

        let d_knee = (solve.knee_position - HIP).length();
        let d_end = (target - solve.knee_position).length();
        assert!((d_knee - UPPER).abs() < 1e-4, "upper length preserved");
        assert!(
            (d_end - LOWER).abs() < 1e-3,
            "lower must land on target within eps, got {d_end}"
        );
    }

    #[test]
    fn knee_displaces_toward_pole_side() {
        let (knee, ankle) = straight_chain();
        let target = Vec3::new(0.0, -1.60, 0.0);
        let solve_fwd = solve_two_bone_ik(HIP, knee, ankle, target, UPPER, LOWER, Vec3::Z).unwrap();
        assert!(solve_fwd.knee_position.z > 1e-3, "bend toward +Z pole");
        assert!(solve_fwd.knee_position.y > target.y);
    }

    #[test]
    fn basic_two_bone_solver_aligns_primary_and_secondary_axes() {
        let hip = Vec3::ZERO;
        let knee = Vec3::X * UPPER;
        let target = Vec3::new(0.66, 0.08, 0.26);
        let pole = Vec3::new(0.15, 0.35, 0.65);
        let solve = solve_basic_two_bone_ik(
            hip,
            Quat::IDENTITY,
            knee,
            target,
            pole,
            UPPER,
            LOWER,
            Vec3::X,
            Vec3::Z,
            1.0,
        )
        .unwrap();
        let upper_axis = (solve.knee_position - hip).normalize();
        let upper_secondary = (solve.knee_position - (solve.effector_position + hip) * 0.5)
            .reject_from_normalized(upper_axis)
            .normalize();
        let lower_axis = (solve.effector_position - solve.knee_position).normalize();
        let lower_secondary = (solve.knee_position - (solve.effector_position + hip) * 0.5)
            .reject_from_normalized(lower_axis)
            .normalize();
        assert!((solve.hip_rotation_world * Vec3::X).dot(upper_axis) > 0.9999);
        assert!((solve.hip_rotation_world * Vec3::Z).dot(upper_secondary) > 0.9999);
        assert!((solve.knee_rotation_world * Vec3::X).dot(lower_axis) > 0.9999);
        assert!((solve.knee_rotation_world * Vec3::Z).dot(lower_secondary) > 0.9999);
    }

    #[test]
    fn basic_two_bone_solver_uses_pole_for_fully_extended_secondary_axes() {
        let hip = Vec3::ZERO;
        let knee = Vec3::X * UPPER;
        let target = Vec3::X * 3.0;
        let pole = Vec3::new(0.0, 0.3, 1.0);
        let solve = solve_basic_two_bone_ik(
            hip,
            Quat::IDENTITY,
            knee,
            target,
            pole,
            UPPER,
            LOWER,
            Vec3::X,
            Vec3::Z,
            1.0,
        )
        .unwrap();
        let upper_pole = pole.reject_from_normalized(Vec3::X).normalize();
        let lower_pole = (pole - solve.knee_position)
            .reject_from_normalized(Vec3::X)
            .normalize();
        assert!((solve.hip_rotation_world * Vec3::X).dot(Vec3::X) > 0.9999);
        assert!((solve.knee_rotation_world * Vec3::X).dot(Vec3::X) > 0.9999);
        assert!((solve.hip_rotation_world * Vec3::Z).dot(upper_pole) > 0.9999);
        assert!((solve.knee_rotation_world * Vec3::Z).dot(lower_pole) > 0.9999);
        assert!((solve.effector_position.x - (UPPER + LOWER)).abs() < 1e-5);
    }

    #[test]
    fn basic_two_bone_solver_preserves_a_translated_pole_location() {
        let local_hip = Vec3::ZERO;
        let local_knee = Vec3::X * UPPER;
        let local_target = Vec3::new(0.64, 0.12, 0.22);
        let local_pole = Vec3::new(0.08, 0.36, 0.58);
        let local = solve_basic_two_bone_ik(
            local_hip,
            Quat::IDENTITY,
            local_knee,
            local_target,
            local_pole,
            UPPER,
            LOWER,
            Vec3::X,
            Vec3::Z,
            1.0,
        )
        .unwrap();
        let translation = Vec3::new(6.0, -1.5, 9.0);
        let translated = solve_basic_two_bone_ik(
            local_hip + translation,
            Quat::IDENTITY,
            local_knee + translation,
            local_target + translation,
            local_pole + translation,
            UPPER,
            LOWER,
            Vec3::X,
            Vec3::Z,
            1.0,
        )
        .unwrap();
        assert!(
            translated
                .knee_position
                .distance(local.knee_position + translation)
                < 1e-5
        );
        assert!(
            translated
                .effector_position
                .distance(local.effector_position + translation)
                < 1e-5
        );
        assert!(
            translated
                .hip_rotation_world
                .dot(local.hip_rotation_world)
                .abs()
                > 0.999_99
        );
        assert!(
            translated
                .knee_rotation_world
                .dot(local.knee_rotation_world)
                .abs()
                > 0.999_99
        );
    }

    #[test]
    fn stable_poles_preserve_mirrored_reference_planes() {
        let left_hip = Vec3::new(-0.2, 0.0, 0.0);
        let left_knee = Vec3::new(-0.2, -0.4, 0.16);
        let left_ankle = Vec3::new(-0.2, -0.8, 0.0);
        let right_hip = Vec3::new(0.2, 0.0, 0.0);
        let right_knee = Vec3::new(0.2, -0.4, -0.16);
        let right_ankle = Vec3::new(0.2, -0.8, 0.0);
        let mut left = StableTwoBonePole::default();
        let mut right = StableTwoBonePole::default();
        let left_direction = left.resolve(
            Quat::IDENTITY,
            left_hip,
            left_knee,
            left_ankle,
            left_ankle + Vec3::new(0.04, 0.0, 0.02),
            Vec3::Z,
        );
        let right_direction = right.resolve(
            Quat::IDENTITY,
            right_hip,
            right_knee,
            right_ankle,
            right_ankle + Vec3::new(-0.04, 0.0, 0.02),
            Vec3::Z,
        );

        assert!(left.is_initialized() && right.is_initialized());
        assert!(left_direction.dot(right_direction) < -0.8);
    }

    #[test]
    fn stable_pole_reuses_the_last_plane_when_the_chain_is_collinear() {
        let hip = Vec3::ZERO;
        let knee = Vec3::new(0.0, -0.4, 0.14);
        let ankle = Vec3::new(0.0, -0.8, 0.0);
        let mut pole = StableTwoBonePole::default();
        let first = pole.resolve(Quat::IDENTITY, hip, knee, ankle, ankle, Vec3::Z);
        let second = pole.resolve(
            Quat::IDENTITY,
            hip,
            Vec3::new(0.0, -0.4, 0.0),
            ankle,
            ankle,
            Vec3::Z,
        );

        assert!(first.dot(second) > 0.999);
        assert!(pole.last_direction_world().dot(first) > 0.999);
    }

    #[test]
    fn rotations_map_current_dirs_onto_solved_dirs() {
        let (knee, ankle) = straight_chain();
        let target = Vec3::new(-0.12, -0.78, 0.08);
        let solve = solve_two_bone_ik(HIP, knee, ankle, target, UPPER, LOWER, Vec3::NEG_Y).unwrap();
        let cur_upper = (knee - HIP).normalize();
        let new_upper = (solve.knee_position - HIP).normalize();
        let rotated = solve.hip_rotation_world * cur_upper;
        assert!((rotated - new_upper).length() < 1e-4);

        let cur_lower = (ankle - knee).normalize();
        let new_lower = (target - solve.knee_position).normalize();
        let rotated_lower = solve.knee_rotation_world * cur_lower;
        assert!((rotated_lower - new_lower).length() < 1e-3);
    }

    #[test]
    fn unreachable_target_clamps_to_full_extension() {
        let (knee, ankle) = straight_chain();
        let far = Vec3::new(0.0, -50.0, 0.0);
        let solve = solve_two_bone_ik(HIP, knee, ankle, far, UPPER, LOWER, Vec3::Z).unwrap();
        let axis = (far - HIP).normalize();
        let expected_knee = HIP + axis * UPPER;
        assert!(
            (solve.knee_position - expected_knee).length() < 0.05,
            "near-straight chain points at target, got {:?}",
            solve.knee_position
        );
    }

    #[test]
    fn over_close_target_clamps_to_inner_limit() {
        let (knee, ankle) = straight_chain();
        let near = Vec3::new(0.0, -0.01, 0.0);
        let solve = solve_two_bone_ik(HIP, knee, ankle, near, UPPER, LOWER, Vec3::Z).unwrap();
        let d = (solve.knee_position - HIP).length();
        assert!((d - UPPER).abs() < 1e-3);
        let end_dist = (near - HIP).length();
        let min_reach = (UPPER - LOWER).abs().max(MIN_DISTANCE);
        assert!(end_dist >= min_reach - 1e-3);
    }

    #[test]
    fn already_satisfied_pose_yields_identity_rotations() {
        let knee = Vec3::new(0.0, -0.40, 0.2131);
        let ankle = knee + Vec3::new(0.0, -0.30, -0.3354).normalize_or_zero() * LOWER;
        let target = ankle;
        let solve = solve_two_bone_ik(HIP, knee, ankle, target, UPPER, LOWER, Vec3::Z).unwrap();
        assert!(
            solve.hip_rotation_world.angle_between(Quat::IDENTITY) < 0.03,
            "hip rot {:?}",
            solve.hip_rotation_world
        );
        assert!(
            solve.knee_rotation_world.angle_between(Quat::IDENTITY) < 0.03,
            "knee rot {:?}",
            solve.knee_rotation_world
        );
    }

    #[test]
    fn pole_parallel_to_axis_does_not_panic_and_still_solves() {
        let (knee, ankle) = straight_chain();
        let target = Vec3::new(0.0, -1.60, 0.0);
        let solve = solve_two_bone_ik(HIP, knee, ankle, target, UPPER, LOWER, Vec3::NEG_Y).unwrap();
        assert!((solve.knee_position - HIP).length() - UPPER < 1e-4);
    }

    #[test]
    fn degenerate_inputs_return_none() {
        let (_, ankle) = straight_chain();
        assert!(
            solve_two_bone_ik(HIP, Vec3::ZERO, ankle, Vec3::ONE, UPPER, LOWER, Vec3::Z).is_none()
        );
        let clamped = solve_two_bone_ik(
            HIP,
            Vec3::Y,
            ankle,
            Vec3::new(0.001, -0.02, 0.0),
            UPPER,
            LOWER,
            Vec3::Z,
        );
        assert!(clamped.is_some(), "inner clamp keeps the chain solvable");
        assert!(
            solve_two_bone_ik(
                HIP,
                Vec3::Y,
                ankle,
                Vec3::new(0.0, -1.0, 0.0),
                0.0,
                LOWER,
                Vec3::Z
            )
            .is_none()
        );
    }

    #[test]
    fn world_to_local_conversion_round_trips() {
        let parent = Quat::from_rotation_y(0.7);
        let world = Quat::from_rotation_x(0.3);
        let local = world_rotation_to_local(parent, world);
        let rebuilt = parent * local;
        assert!(rebuilt.angle_between(world) < 1e-5);
    }

    #[test]
    fn rotation_weight_blends_continuously_between_animation_and_ik() {
        let animation = Quat::from_rotation_y(-0.4);
        let ik = Quat::from_rotation_y(0.8);
        let zero = blend_ik_rotation(animation, ik, 0.0);
        let half = blend_ik_rotation(animation, ik, 0.5);
        let full = blend_ik_rotation(animation, ik, 1.0);

        assert!(zero.dot(animation).abs() > 0.999_99);
        assert!(full.dot(ik).abs() > 0.999_99);
        assert!(half.angle_between(animation) > 0.1);
        assert!(half.angle_between(ik) > 0.1);
    }

    #[test]
    fn child_rotation_compensates_for_the_parent_solve_before_reaching_target_axis() {
        let parent_delta = Quat::from_rotation_z(0.35);
        let child_rotation = Quat::IDENTITY;
        let local_axis = Vec3::NEG_Y;
        let target_axis = Vec3::new(0.25, -0.9, 0.35).normalize();
        let corrected = child_world_rotation_after_parent_rotation(
            parent_delta,
            child_rotation,
            local_axis,
            target_axis,
        )
        .unwrap();

        let actual_axis = corrected * local_axis;
        assert!(actual_axis.dot(target_axis) > 0.999_99);
    }
}
