# Bevy Control Rig

A Bevy 0.20 plugin for executing rig programs on skeletal hierarchies. It provides typed registers, local and global bone transforms, inverse kinematics, and spring and damping units.

```toml
[dependencies]
bevy_control_rig = { git = "https://github.com/millionart/bevy_control_rig" }
```

```rust
use bevy::prelude::*;
use bevy_control_rig::ControlRigPlugin;

fn main() {
    App::new().add_plugins(ControlRigPlugin);
}
```

`ControlRigPlugin` registers `RigDefinition` and its reflected dependencies. Deserialize a definition with Serde, create a `RigHierarchy` and registers for each rig instance, and run an entry with `RigProgram::execute_in_units`. `RigUnitContext` runs the built-in units with the current curves, time, and optional line trace. The program's unit callback can dispatch custom operations. See `tests/plugin.rs` for a runnable example.

## Bevy Animation Graph

Use Control Rig alongside Bevy Animation Graph, following the separation between Control Rig and animation graphs in Unreal Engine. A graph node provides its input pose and curves, evaluates the rig at the appropriate lifecycle stage, and returns the edited pose. The same rig can run directly in a Bevy system. The compatible Bevy 0.20.1 revision is [Bevy Animation Graph 0.11.0](https://github.com/millionart/bevy_animation_graph/tree/6d706f61c35a6cf7b1c96430a56ae46241b4d51f).

The rig owns its mutable bone state and registers per instance. The caller supplies the frame's input pose and decides when to run construction and forward solve. Use the definition's `meters_per_unit` when binding registers and executing a program. A caller that consumes serialized unit names can map them to the identifiers accepted by `RigUnitContext::call`.

Licensed under either MIT or Apache-2.0 at your option.
