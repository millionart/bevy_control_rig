use bevy::prelude::{GlobalTransform, Quat, Vec3};

#[cfg(test)]
use bevy::prelude::{EulerRot, Transform};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorldPoint(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArmaturePoint(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ParentPoint(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoneLocalPoint(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorldDirection(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArmatureDirection(pub Vec3);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorldRotation(pub Quat);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArmatureRotation(pub Quat);

fn normalized(direction: Vec3) -> Vec3 {
    direction.try_normalize().unwrap_or(Vec3::ZERO)
}

pub fn armature_point_from_world(
    armature_world: &GlobalTransform,
    point: WorldPoint,
) -> ArmaturePoint {
    ArmaturePoint(
        armature_world
            .to_matrix()
            .inverse()
            .transform_point3(point.0),
    )
}

pub fn world_point_from_armature(
    armature_world: &GlobalTransform,
    point: ArmaturePoint,
) -> WorldPoint {
    WorldPoint(armature_world.to_matrix().transform_point3(point.0))
}

pub fn parent_point_from_world(parent_world: &GlobalTransform, point: WorldPoint) -> ParentPoint {
    ParentPoint(parent_world.to_matrix().inverse().transform_point3(point.0))
}

pub fn world_point_from_parent(parent_world: &GlobalTransform, point: ParentPoint) -> WorldPoint {
    WorldPoint(parent_world.to_matrix().transform_point3(point.0))
}

pub fn bone_local_point_from_world(
    parent_world: &GlobalTransform,
    point: WorldPoint,
) -> BoneLocalPoint {
    BoneLocalPoint(parent_world.to_matrix().inverse().transform_point3(point.0))
}

pub fn world_point_from_bone_local(
    parent_world: &GlobalTransform,
    point: BoneLocalPoint,
) -> WorldPoint {
    WorldPoint(parent_world.to_matrix().transform_point3(point.0))
}

pub fn armature_direction_from_world(
    armature_world: &GlobalTransform,
    direction: WorldDirection,
) -> ArmatureDirection {
    ArmatureDirection(normalized(
        armature_world.rotation().inverse() * direction.0,
    ))
}

pub fn world_direction_from_armature(
    armature_world: &GlobalTransform,
    direction: ArmatureDirection,
) -> WorldDirection {
    WorldDirection(normalized(armature_world.rotation() * direction.0))
}

pub fn armature_rotation_from_world(
    armature_world: &GlobalTransform,
    rotation: WorldRotation,
) -> ArmatureRotation {
    ArmatureRotation((armature_world.rotation().inverse() * rotation.0).normalize())
}

pub fn world_rotation_from_armature(
    armature_world: &GlobalTransform,
    rotation: ArmatureRotation,
) -> WorldRotation {
    WorldRotation((armature_world.rotation() * rotation.0).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(translation: Vec3, rotation: Quat) -> GlobalTransform {
        GlobalTransform::from(Transform::from_translation(translation).with_rotation(rotation))
    }

    #[test]
    fn points_and_directions_round_trip_under_rotation() {
        let armature = frame(
            Vec3::new(3.0, -2.0, 1.5),
            Quat::from_euler(EulerRot::YXZ, 0.8, -0.3, 0.25),
        );
        let world_point = WorldPoint(Vec3::new(-1.2, 0.4, 5.5));
        let armature_point = armature_point_from_world(&armature, world_point);
        let rebuilt_point = world_point_from_armature(&armature, armature_point);
        let world_direction = WorldDirection(Vec3::new(0.3, 0.8, -0.45).normalize());
        let armature_direction = armature_direction_from_world(&armature, world_direction);
        let rebuilt_direction = world_direction_from_armature(&armature, armature_direction);

        assert!(rebuilt_point.0.distance(world_point.0) < 1e-5);
        assert!(rebuilt_direction.0.dot(world_direction.0) > 0.999_99);

        let world_rotation = WorldRotation(Quat::from_euler(EulerRot::YXZ, -0.4, 0.2, 0.7));
        let armature_rotation = armature_rotation_from_world(&armature, world_rotation);
        let rebuilt_rotation = world_rotation_from_armature(&armature, armature_rotation);
        assert!(rebuilt_rotation.0.dot(world_rotation.0).abs() > 0.999_99);
    }

    #[test]
    fn parent_and_bone_local_frames_remain_instance_isolated() {
        let parent_a = frame(Vec3::new(2.0, 1.0, -3.0), Quat::from_rotation_y(0.5));
        let parent_b = frame(Vec3::new(-4.0, 0.5, 2.0), Quat::from_rotation_y(-0.8));
        let world = WorldPoint(Vec3::new(1.2, -0.4, 0.75));
        let parent_local_a = parent_point_from_world(&parent_a, world);
        let bone_local_b = bone_local_point_from_world(&parent_b, world);

        assert!(
            world_point_from_parent(&parent_a, parent_local_a)
                .0
                .distance(world.0)
                < 1e-5
        );
        assert!(
            world_point_from_bone_local(&parent_b, bone_local_b)
                .0
                .distance(world.0)
                < 1e-5
        );
        assert!(parent_local_a.0.distance(bone_local_b.0) > 0.1);
    }
}
