use bevy::prelude::{App, Plugin};

pub mod hierarchy;
pub mod ik;
pub mod ik_space;
pub mod math;
pub mod program;
pub mod rig_ik;
pub mod units;

pub struct ControlRigPlugin;

impl Plugin for ControlRigPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<program::RigDefinition>();
    }
}
