//! Interface strings: one key, one French and one English text (`locales/fr.json`,
//! `locales/en.json`, embedded). A view never holds a literal label; `tests/ui_i18n.rs`
//! checks that both files carry the same keys and that every key a source file asks for
//! exists.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt::Display;
use std::sync::OnceLock;

use crate::settings::Lang;

pub const FR: &str = include_str!("locales/fr.json");
pub const EN: &str = include_str!("locales/en.json");

thread_local! {
    // The interface runs on one thread; each test thread gets its own language.
    static LANG: Cell<Lang> = const { Cell::new(Lang::Fr) };
}

pub fn set_lang(lang: Lang) {
    LANG.with(|l| l.set(lang));
}

pub fn lang() -> Lang {
    LANG.with(Cell::get)
}

/// The table of one language. A locale file that does not parse is a build defect: the
/// parity test catches it, and the interface then shows the keys.
pub fn table(lang: Lang) -> &'static BTreeMap<String, String> {
    static TABLES: OnceLock<[BTreeMap<String, String>; 2]> = OnceLock::new();
    let tables =
        TABLES.get_or_init(|| [FR, EN].map(|json| serde_json::from_str(json).unwrap_or_default()));
    match lang {
        Lang::Fr => &tables[0],
        Lang::En => &tables[1],
    }
}

/// The text of `key` in the current language; the key itself when it is missing.
pub fn t(key: &'static str) -> &'static str {
    table(lang()).get(key).map_or(key, String::as_str)
}

/// [`t`] with `{name}` placeholders filled in.
pub fn tf(key: &'static str, args: &[(&str, &dyn Display)]) -> String {
    let mut text = t(key).to_owned();
    for (name, value) in args {
        text = text.replace(&format!("{{{name}}}"), &value.to_string());
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_reads_in_the_current_language_and_placeholders_are_filled() {
        set_lang(Lang::Fr);
        assert_eq!(t("tab.analyses"), "Analyses");
        assert_eq!(t("tab.graph"), "Graphe 3D");
        set_lang(Lang::En);
        assert_eq!(t("tab.graph"), "3D Graph");
        assert_eq!(t("no.such.key"), "no.such.key");
        let line = tf("tree.count", &[("shown", &3), ("total", &12)]);
        assert!(line.contains('3') && line.contains("12"), "{line}");
    }
}
