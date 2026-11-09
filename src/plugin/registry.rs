//! The set of extraction methods a program knows: the built-in ones, then its own.

use std::collections::BTreeMap;

use serde::Serialize;

use super::builtin::Builtin;
use super::{
    is_valid_id, Catalog, ExtensionPoint, ExtractionMethod, Implementation, PluginError,
    API_VERSION, EXTRACTION_METHOD_POINT,
};

/// Where an implementation comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Part of the engine.
    Builtin,
    /// Added with [`Registry::register`].
    Registered,
}

impl Origin {
    /// `builtin` or `registered`, as serialized.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Registered => "registered",
        }
    }
}

/// Extraction methods by identifier. Listing order is deterministic: the built-in methods
/// in their historical order (`mean_diff`, `svd`, `leace`, the order of the CLI's possible
/// values and of the MCP schema), then the registered ones by identifier.
pub struct Registry {
    builtin: Vec<Box<dyn ExtractionMethod>>,
    registered: BTreeMap<&'static str, Box<dyn ExtractionMethod>>,
}

impl Registry {
    /// The three built-in methods, nothing else.
    pub fn with_builtin() -> Self {
        Self {
            builtin: Builtin::all()
                .into_iter()
                .map(|b| Box::new(b) as Box<dyn ExtractionMethod>)
                .collect(),
            registered: BTreeMap::new(),
        }
    }

    /// Adds a method. Refuses an identifier that breaks `[a-z][a-z0-9_]{1,31}`, that of a
    /// built-in method, or one already registered; the registry is unchanged then.
    pub fn register(&mut self, method: Box<dyn ExtractionMethod>) -> Result<(), PluginError> {
        let id = method.id();
        if !is_valid_id(id) {
            return Err(PluginError::InvalidId(id.to_owned()));
        }
        if self.builtin.iter().any(|b| b.id() == id) {
            return Err(PluginError::ReservedId(id.to_owned()));
        }
        if self.registered.contains_key(id) {
            return Err(PluginError::DuplicateId(id.to_owned()));
        }
        self.registered.insert(id, method);
        Ok(())
    }

    /// The method with this identifier.
    pub fn get(&self, id: &str) -> Option<&dyn ExtractionMethod> {
        self.list().find(|(m, _)| m.id() == id).map(|(m, _)| m)
    }

    /// Every method with its origin, in listing order.
    pub fn list(&self) -> impl Iterator<Item = (&dyn ExtractionMethod, Origin)> {
        let builtin = self.builtin.iter().map(|m| (m.as_ref(), Origin::Builtin));
        builtin.chain(
            self.registered
                .values()
                .map(|m| (m.as_ref(), Origin::Registered)),
        )
    }

    /// Identifiers in listing order.
    pub fn ids(&self) -> Vec<&'static str> {
        self.list().map(|(m, _)| m.id()).collect()
    }

    /// What `surgeon plugins list` prints.
    pub fn catalog(&self) -> Catalog {
        let implementations = self
            .list()
            .map(|(m, origin)| Implementation {
                id: m.id(),
                origin,
                info: m.describe(),
            })
            .collect();
        Catalog {
            api_version: API_VERSION,
            extension_points: vec![ExtensionPoint {
                id: EXTRACTION_METHOD_POINT,
                rust_trait: "weight_surgeon::plugin::ExtractionMethod",
                implementations,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::plugin::{ActivationSet, ExtractContext, ExtractionOutput, MethodInfo};

    struct Named(&'static str);

    impl ExtractionMethod for Named {
        fn id(&self) -> &'static str {
            self.0
        }
        fn describe(&self) -> MethodInfo {
            MethodInfo {
                title: "test",
                summary: "test",
                params: vec![],
                requires_paired: false,
                sidecars: vec![],
            }
        }
        fn extract(
            &self,
            _: &ExtractContext,
            _: &ActivationSet,
            _: &AtomicBool,
        ) -> Result<ExtractionOutput, PluginError> {
            Ok(ExtractionOutput::new(vec![1.0]))
        }
    }

    #[test]
    fn builtin_first_then_registered_by_id_whatever_the_order() {
        let mut registry = Registry::with_builtin();
        assert_eq!(registry.ids(), ["mean_diff", "svd", "leace"]);
        for id in ["zeta", "alpha", "mid_2"] {
            registry.register(Box::new(Named(id))).unwrap();
        }
        assert_eq!(
            registry.ids(),
            ["mean_diff", "svd", "leace", "alpha", "mid_2", "zeta"]
        );
        let origins: Vec<Origin> = registry.list().map(|(_, o)| o).collect();
        assert_eq!(origins[2..4], [Origin::Builtin, Origin::Registered]);
        assert_eq!(registry.get("alpha").map(|m| m.id()), Some("alpha"));
        assert!(registry.get("pca").is_none());
    }

    #[test]
    fn bad_reserved_and_duplicate_ids_are_refused_without_change() {
        let mut registry = Registry::with_builtin();
        registry.register(Box::new(Named("median_diff"))).unwrap();
        for (id, expected) in [
            ("Median", "InvalidId"),
            ("m", "InvalidId"),
            ("svd", "ReservedId"),
            ("mean_diff", "ReservedId"),
            ("median_diff", "DuplicateId"),
        ] {
            let err = registry.register(Box::new(Named(id))).unwrap_err();
            assert!(format!("{err:?}").starts_with(expected), "{id}: {err:?}");
        }
        assert_eq!(registry.ids().len(), 4);
    }
}
