//! Target model resolution shared by `.surgery` apply and `lora merge` (single file or
//! sharded directory: which shard holds which tensor, which shards to back up).
//! Body written by the `lowrank` agent; the `lora` agent (wave 2) reuses it.
