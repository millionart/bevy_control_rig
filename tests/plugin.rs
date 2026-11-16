use bevy::{math::DVec3, prelude::*};
use bevy_control_rig::{
    ControlRigPlugin,
    hierarchy::{RigHierarchy, RigTransform},
    math::RigValue,
    program::{RigDefinition, RigInstruction, RigProgram, RigValueRef},
    rig_ik::RigGeometry,
    units::RigUnitContext,
};

#[test]
fn plugin_registers_definition_and_runs_a_program() {
    let mut app = App::new();
    app.add_plugins(ControlRigPlugin);
    let registry = app.world().resource::<AppTypeRegistry>().read();
    assert!(
        registry
            .get(std::any::TypeId::of::<RigDefinition>())
            .is_some()
    );
    assert!(registry.get(std::any::TypeId::of::<RigValue>()).is_some());

    let definition = RigDefinition {
        program: RigProgram {
            initial_values: vec![
                RigValue::Name("Weight".into()),
                RigValue::Bool(false),
                RigValue::F32(0.0),
                RigValue::List(vec![]),
            ],
            literal_count: 1,
            entries: [("Forwards Solve".into(), 0)].into(),
            instructions: vec![
                RigInstruction::Call {
                    function: "FRigUnit_GetCurveValue::Execute".into(),
                    arguments: (0..4)
                        .map(|register| RigValueRef {
                            register,
                            path: vec![],
                        })
                        .collect(),
                    outputs: vec![1, 2],
                },
                RigInstruction::Exit,
            ],
        },
        externals: default(),
        bones: vec![],
        curves: default(),
        translation_tolerance: 0.0,
        geometry: RigGeometry {
            meters_per_unit: 1.0,
            near_zero: 1e-4,
            normalize_squared: 1e-8,
            minimum_length: 1e-8,
            solve_near_zero: 1e-4,
            basis: [DVec3::X, DVec3::Y, DVec3::Z],
        },
    };
    let encoded = serde_json::to_string(&definition).expect("serialize rig definition");
    let definition: RigDefinition =
        serde_json::from_str(&encoded).expect("deserialize rig definition");
    let mut hierarchy = RigHierarchy::new(definition.bones, definition.translation_tolerance)
        .expect("initialize rig hierarchy");
    let mut registers = definition.program.initial_values.clone();
    let mut context = RigUnitContext::<String> {
        hierarchy: &mut hierarchy,
        curves: &[("weight".into(), 0.75)].into(),
        game_world: true,
        construction: false,
        delta_seconds: 1.0 / 60.0,
        geometry: &definition.geometry,
        to_world: RigTransform::default(),
        line_trace: None,
    };
    definition
        .program
        .execute_in_units(
            "Forwards Solve",
            &mut registers,
            definition.geometry.meters_per_unit,
            |_, name, args| context.call(name, args),
        )
        .expect("run rig program");
    assert_eq!(registers[1..3], [RigValue::Bool(true), RigValue::F32(0.75)]);
}
